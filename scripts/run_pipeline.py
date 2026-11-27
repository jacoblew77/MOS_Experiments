#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from fgz_nn_lab.data_ingest import (
    fetch_hourly_predictors_for_stations,
    fetch_iem_mos_bulletins,
    fetch_live_operational_inputs,
    fetch_unestimated_iem_observations,
    load_station_catalog,
)
from fgz_nn_lab.evaluate import (
    compute_metrics_bundle,
    log_experiment_run,
    write_single_file_master_bundle,
)
from fgz_nn_lab.features import (
    TensorBundle,
    build_aligned_tensors,
    build_live_inference_bundle,
)
from fgz_nn_lab.models import (
    apply_blend_weights,
    fit_l1_optimal_blend_weights,
    fit_predict_lgbm_baseline,
    fit_predict_ridge_baseline,
    predict_lgbm_grid,
    train_and_predict_nn,
)
from fgz_nn_lab.splits import FoldScaler, make_purged_expanding_splits


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def pretrain_era5_backbone(
    stations_active: pd.DataFrame,
    raw_dir: Path,
    cfg: dict[str, Any],
    target_cutoff_date: np.datetime64,
    vocab: list[str],
    fast_smoke: bool = False,
) -> dict[str, Any]:
    e_start = cfg["dates"]["smoke_start"] if fast_smoke else cfg["dates"]["era5_start"]
    e_end = cfg["dates"]["smoke_end"] if fast_smoke else cfg["dates"]["era5_end"]

    print(
        f"    [{_ts()}] [PRETRAIN] Loading 100%-real observations + ERA5 ({e_start}..{e_end}, cutoff < {target_cutoff_date})...",
        flush=True,
    )
    obs_e, _, _ = fetch_unestimated_iem_observations(
        stations_active, e_start, e_end, raw_dir,
        min_clean_days=120 if fast_smoke else 400,
        min_recent_days=50,
        max_coop=20, max_asos=20,
    )
    hourly_e = fetch_hourly_predictors_for_stations(stations_active, e_start, e_end, raw_dir, source_mode="era5")

    qg = cfg["qc_and_gating"]
    ebundle = build_aligned_tensors(
        stations_df=stations_active,
        coop_df=obs_e,
        hourly_by_source=hourly_e,
        issue_hour_utc=int(qg["issue_hour_utc"]),
        morning_min_gate_utc=int(qg["morning_min_gate_utc"]),
        afternoon_max_gate_utc=int(qg["afternoon_max_gate_utc"]),
        min_causal_lag_hours=int(qg["min_causal_lag_hours"]),
        station_vocab=vocab,
    )

    safe_cutoff = target_cutoff_date - np.timedelta64(int(cfg["splits"]["purge_days"]), "D")
    elig = np.where(ebundle.issue_dates < safe_cutoff)[0]
    if len(elig) < 180:
        return {}

    split_pt = int(len(elig) * 0.88)
    tr, va = elig[:split_pt], elig[split_pt:]
    scaler = FoldScaler(min_valid_samples=80).fit(
        ebundle.x_dyn[tr], ebundle.mask_dyn[tr], ebundle.x_ctx[tr],
        ebundle.raw_anchor[tr], ebundle.y_true[tr], ebundle.station_ids[tr],
        ebundle.raw_anchor_candidates[tr],
    )
    xd_tr, mk_tr, xc_tr = scaler.transform_inputs(ebundle.x_dyn[tr], ebundle.mask_dyn[tr], ebundle.x_ctx[tr])
    xd_va, mk_va, xc_va = scaler.transform_inputs(ebundle.x_dyn[va], ebundle.mask_dyn[va], ebundle.x_ctx[va])
    anc_tr = scaler.calibrate_anchor(ebundle.raw_anchor[tr], ebundle.station_ids[tr], ebundle.raw_anchor_candidates[tr])
    anc_va = scaler.calibrate_anchor(ebundle.raw_anchor[va], ebundle.station_ids[va], ebundle.raw_anchor_candidates[va])

    _, _, pre_model = train_and_predict_nn(
        xd_tr, mk_tr, xc_tr, ebundle.station_indices[tr], anc_tr, ebundle.y_true[tr],
        xd_va, mk_va, xc_va, ebundle.station_indices[va], anc_va, ebundle.y_true[va],
        xd_va, mk_va, xc_va, ebundle.station_indices[va], anc_va,
        nn_cfg=cfg["nn"],
        num_stations=len(vocab),
        seed=int(cfg["seed"]),
        fold_label="era5_pretrain",
        override_epochs=10 if fast_smoke else int(cfg["nn"].get("pretrain_epochs", 28)),
    )
    return {k: v.cpu() for k, v in pre_model.state_dict().items()}


def main() -> None:
    run_t0 = time.monotonic()
    parser = argparse.ArgumentParser(description="One-Click Operational FGZ COOP + ASOS AI-MOS Suite")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--stations", type=Path, default=Path("configs/stations_fgz_coop.json"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    st_cfg = json.loads(args.stations.read_text(encoding="utf-8")) if False else yaml.safe_load(args.stations.read_text(encoding="utf-8"))
    raw_dir = Path("data/raw")
    runs_dir = Path("data/runs")

    candidates_df = load_station_catalog(args.stations, raw_dir=None if args.fast_smoke else raw_dir)
    qg = cfg["qc_and_gating"]

    if args.fast_smoke:
        candidates_df = candidates_df.iloc[:4].reset_index(drop=True)
        start_date, end_date = cfg["dates"]["smoke_start"], cfg["dates"]["smoke_end"]
        min_clean, min_recent = 120, 60
        cfg["nn"]["max_epochs"] = 10
    else:
        start_date, end_date = cfg["dates"]["true_lead_start"], cfg["dates"]["true_lead_end"]
        min_clean = int(qg.get("min_clean_obs_days_lead", 380))
        min_recent = int(qg.get("min_recent_year_days", 150))

    print(f"\n[{_ts()}] >>> [1/5] DISCOVERING FGZ ASOS + COOP STATIONS & UN-ESTIMATED OBSERVATIONS <<<", flush=True)
    obs_df, audit_df, stations_active = fetch_unestimated_iem_observations(
        candidates_df, start_date, end_date, raw_dir,
        min_clean_days=min_clean,
        min_recent_days=min_recent,
        max_coop=int(st_cfg.get("max_coop_stations", 12)),
        max_asos=int(st_cfg.get("max_asos_stations", 8)),
    )
    vocab = sorted(stations_active["station_id"].tolist())

    print(f"\n[{_ts()}] >>> [2/5] INGESTING OPERATIONAL NWP, 2.5KM NBM GRIDDED MOS & IEM MOS BULLETINS <<<", flush=True)
    mos_df = fetch_iem_mos_bulletins(stations_active, start_date, end_date, raw_dir)
    hourly_by_src = fetch_hourly_predictors_for_stations(
        stations_active, start_date, end_date, raw_dir, source_mode="true_lead_nwp"
    )

    print(f"\n[{_ts()}] >>> [3/5] ALIGNING 5-DAY MULTI-MODEL & MOS TENSORS <<<", flush=True)
    bundle = build_aligned_tensors(
        stations_df=stations_active,
        coop_df=obs_df,
        hourly_by_source=hourly_by_src,
        mos_df=mos_df,
        issue_hour_utc=int(qg["issue_hour_utc"]),
        morning_min_gate_utc=int(qg["morning_min_gate_utc"]),
        afternoon_max_gate_utc=int(qg["afternoon_max_gate_utc"]),
        min_causal_lag_hours=int(qg["min_causal_lag_hours"]),
        station_vocab=vocab,
    )

    cov_df = pd.DataFrame(
        [
            {
                "channel": ch_name,
                "valid_frac_day1": round(float(bundle.mask_dyn[:, 0, i].mean()), 4),
                "valid_frac_day5": round(float(bundle.mask_dyn[:, 4, i].mean()), 4),
            }
            for i, ch_name in enumerate(bundle.dyn_feature_names)
        ]
    )

    sp = cfg["splits"]
    folds = make_purged_expanding_splits(
        bundle.issue_dates,
        test_days=90 if args.fast_smoke else int(sp["test_days"]),
        val_folds=1 if args.fast_smoke else int(sp["val_folds"]),
        val_fold_days=90 if args.fast_smoke else int(sp["val_fold_days"]),
        purge_days=int(sp["purge_days"]),
    )

    print(f"\n[{_ts()}] >>> [4/5] TRAINING BASELINES, ERA5-PRETRAINED NN & OPERATIONAL NN+LGBM STACK <<<", flush=True)
    test_idx = folds[-1].test_idx
    n_folds = len(folds)

    preds_val_by_mod: dict[str, list[np.ndarray]] = {k: [] for k in ("anchor", "ridge", "lgbm", "nn_scratch", "nn_era5_pretrained", "nn_lgbm_stack")}
    preds_test_ens: dict[str, np.ndarray] = {k: np.zeros((len(test_idx), 5, 2), dtype=np.float32) for k in preds_val_by_mod}
    val_y_all, val_sid_all, val_tier_all, val_snow_all, val_net_all = [], [], [], [], []
    fold_diag: dict[str, Any] = {}

    last_fold_artifacts: dict[str, Any] = {}

    for f_i, fold in enumerate(folds, start=1):
        tr, va, te = fold.train_idx, fold.val_idx, fold.test_idx
        print(
            f"\n  [{_ts()}] === Fold {fold.fold_name} ({f_i}/{n_folds}) | "
            f"Train[{bundle.issue_dates[tr[0]]}..{bundle.issue_dates[tr[-1]]}] "
            f"Val[{bundle.issue_dates[va[0]]}..{bundle.issue_dates[va[-1]]}] "
            f"Test[{bundle.issue_dates[te[0]]}..{bundle.issue_dates[te[-1]]}] ===",
            flush=True,
        )
        scaler = FoldScaler(
            min_valid_samples=30 if len(tr) < 500 else int(sp.get("min_valid_channel_samples", 120)),
            clip_zscore=float(sp.get("clip_zscore", 8.0)),
        ).fit(
            bundle.x_dyn[tr], bundle.mask_dyn[tr], bundle.x_ctx[tr],
            bundle.raw_anchor[tr], bundle.y_true[tr], bundle.station_ids[tr],
            bundle.raw_anchor_candidates[tr],
        )

        xd_tr, mk_tr, xc_tr = scaler.transform_inputs(bundle.x_dyn[tr], bundle.mask_dyn[tr], bundle.x_ctx[tr])
        xd_va, mk_va, xc_va = scaler.transform_inputs(bundle.x_dyn[va], bundle.mask_dyn[va], bundle.x_ctx[va])
        xd_te, mk_te, xc_te = scaler.transform_inputs(bundle.x_dyn[te], bundle.mask_dyn[te], bundle.x_ctx[te])

        anc_tr = scaler.calibrate_anchor(bundle.raw_anchor[tr], bundle.station_ids[tr], bundle.raw_anchor_candidates[tr])
        anc_va = scaler.calibrate_anchor(bundle.raw_anchor[va], bundle.station_ids[va], bundle.raw_anchor_candidates[va])
        anc_te = scaler.calibrate_anchor(bundle.raw_anchor[te], bundle.station_ids[te], bundle.raw_anchor_candidates[te])

        # 1. Anchor & Ridge
        vp_anc, tp_anc = anc_va, anc_te
        vp_rid = fit_predict_ridge_baseline(xd_tr, mk_tr, xc_tr, anc_tr, bundle.y_true[tr], xd_va, mk_va, xc_va, anc_va)
        tp_rid = fit_predict_ridge_baseline(xd_tr, mk_tr, xc_tr, anc_tr, bundle.y_true[tr], xd_te, mk_te, xc_te, anc_te)

        # 2. LightGBM
        vp_lgb, tp_lgb, lgb_models = fit_predict_lgbm_baseline(
            xd_tr, mk_tr, xc_tr, bundle.station_indices[tr], anc_tr, bundle.y_true[tr],
            xd_va, mk_va, xc_va, bundle.station_indices[va], anc_va, bundle.y_true[va],
            xd_te, mk_te, xc_te, bundle.station_indices[te], anc_te,
            seed=int(cfg["seed"]), fold_label=fold.fold_name,
        )

        # 3. Scratch NN
        vp_nns, tp_nns, _ = train_and_predict_nn(
            xd_tr, mk_tr, xc_tr, bundle.station_indices[tr], anc_tr, bundle.y_true[tr],
            xd_va, mk_va, xc_va, bundle.station_indices[va], anc_va, bundle.y_true[va],
            xd_te, mk_te, xc_te, bundle.station_indices[te], anc_te,
            nn_cfg=cfg["nn"], num_stations=len(vocab), seed=int(cfg["seed"]),
            fold_label=f"{fold.fold_name}_scratch",
        )

        # 4. Two-stage ERA5-Pretrained -> NWP+NBM+MOS Fine-Tuned NN
        pre_state = pretrain_era5_backbone(
            stations_active=stations_active, raw_dir=raw_dir, cfg=cfg,
            target_cutoff_date=bundle.issue_dates[va[0]], vocab=vocab, fast_smoke=args.fast_smoke,
        )
        vp_nnx, tp_nnx, nnx_model = train_and_predict_nn(
            xd_tr, mk_tr, xc_tr, bundle.station_indices[tr], anc_tr, bundle.y_true[tr],
            xd_va, mk_va, xc_va, bundle.station_indices[va], anc_va, bundle.y_true[va],
            xd_te, mk_te, xc_te, bundle.station_indices[te], anc_te,
            nn_cfg=cfg["nn"], num_stations=len(vocab), seed=int(cfg["seed"]),
            fold_label=f"{fold.fold_name}_era5_xfer", pretrained_state_dict=pre_state if pre_state else None,
        )

        # 5. Out-of-Fold Validation-Optimal L1 Stack (ERA5-Pretrained NN + LightGBM)
        blend_w = fit_l1_optimal_blend_weights(vp_nnx, vp_lgb, bundle.y_true[va])
        vp_stk = apply_blend_weights(vp_nnx, vp_lgb, blend_w)
        tp_stk = apply_blend_weights(tp_nnx, tp_lgb, blend_w)

        mode_names_min = {0: "cal", 1: "morn24", 2: "eve24"}
        mode_names_max = {3: "cal", 4: "morn24", 5: "eve24"}
        fold_diag[fold.fold_name] = {
            "active_channels": int(scaler.active_channel_mask.sum()) if scaler.active_channel_mask is not None else 0,
            "total_channels": int(bundle.x_dyn.shape[2]),
            "station_window_modes": {
                s: f"min={mode_names_min.get(scaler.station_min_mode.get(s, 0))},max={mode_names_max.get(scaler.station_max_mode.get(s, 3))}"
                for s in sorted(np.unique(bundle.station_ids[tr]))
            },
            "nn_weight_in_stack_min_d1_5": [round(float(x), 2) for x in blend_w[:, 0]],
            "nn_weight_in_stack_max_d1_5": [round(float(x), 2) for x in blend_w[:, 1]],
        }

        for m_key, vp_arr, tp_arr in [
            ("anchor", vp_anc, tp_anc),
            ("ridge", vp_rid, tp_rid),
            ("lgbm", vp_lgb, tp_lgb),
            ("nn_scratch", vp_nns, tp_nns),
            ("nn_era5_pretrained", vp_nnx, tp_nnx),
            ("nn_lgbm_stack", vp_stk, tp_stk),
        ]:
            preds_val_by_mod[m_key].append(vp_arr)
            preds_test_ens[m_key] += tp_arr / n_folds

        val_y_all.append(bundle.y_true[va])
        val_sid_all.append(bundle.station_ids[va])
        val_tier_all.append(bundle.tiers[va])
        val_snow_all.append(bundle.snow_flag[va])
        val_net_all.append(bundle.networks[va])

        last_fold_artifacts = {
            "scaler": scaler,
            "nn_model": nnx_model,
            "lgb_models": lgb_models,
            "blend_w": blend_w,
        }

    vy_cat = np.concatenate(val_y_all, axis=0)
    vsid_cat = np.concatenate(val_sid_all, axis=0)
    vtier_cat = np.concatenate(val_tier_all, axis=0)
    vsnow_cat = np.concatenate(val_snow_all, axis=0)
    vnet_cat = np.concatenate(val_net_all, axis=0)

    print(f"\n[{_ts()}] >>> OPERATIONAL SUITE TEST LEADERBOARD (100% REAL OBS, TRUE 24h..120h LEAD) <<<", flush=True)
    for m_key in ("anchor", "ridge", "lgbm", "nn_scratch", "nn_era5_pretrained", "nn_lgbm_stack"):
        v_sum, _ = compute_metrics_bundle(
            np.concatenate(preds_val_by_mod[m_key], axis=0), vy_cat, vsid_cat, vtier_cat, vsnow_cat, vnet_cat
        )
        t_sum, t_strat = compute_metrics_bundle(
            preds_test_ens[m_key], bundle.y_true[test_idx], bundle.station_ids[test_idx],
            bundle.tiers[test_idx], bundle.snow_flag[test_idx], bundle.networks[test_idx],
        )
        log_experiment_run(
            runs_dir, run_id=f"op_v3_{m_key}", model_family=m_key,
            feature_tag="nbm_mos_upper_real_obs", val_summary=v_sum, test_summary=t_sum,
            stratified_df=t_strat, config_snapshot=cfg, fold_diagnostics=fold_diag,
        )
        print(
            f"  {m_key.upper():20s} | Test Min MAE: {t_sum['mae_min_1_5']:.3f}F | "
            f"Test Max MAE: {t_sum['mae_max_1_5']:.3f}F | Combined: {t_sum['mae_all_1_5']:.3f}F",
            flush=True,
        )

    print(f"\n[{_ts()}] >>> [5/5] GENERATING LIVE 5-DAY OPERATIONAL FORECAST FOR FGZ STATIONS <<<", flush=True)
    live_records: list[dict[str, Any]] = []
    try:
        live_hourly, live_mos = fetch_live_operational_inputs(stations_active)
        live_bundle, fcast_dates = build_live_inference_bundle(
            stations_active, obs_df, live_hourly, live_mos, vocab,
            int(qg["issue_hour_utc"]), int(qg["morning_min_gate_utc"]),
            int(qg["afternoon_max_gate_utc"]), int(qg["min_causal_lag_hours"]),
        )
        if live_bundle is not None and last_fold_artifacts:
            sc: FoldScaler = last_fold_artifacts["scaler"]
            nn_m = last_fold_artifacts["nn_model"]
            lgb_m = last_fold_artifacts["lgb_models"]
            bw = last_fold_artifacts["blend_w"]

            xd_l, mk_l, xc_l = sc.transform_inputs(live_bundle.x_dyn, live_bundle.mask_dyn, live_bundle.x_ctx)
            anc_l = sc.calibrate_anchor(live_bundle.raw_anchor, live_bundle.station_ids, live_bundle.raw_anchor_candidates)

            device = next(nn_m.parameters()).device
            nn_m.eval()
            with torch.no_grad():
                p_nn = nn_m(
                    torch.from_numpy(xd_l).to(device),
                    torch.from_numpy(mk_l).to(device),
                    torch.from_numpy(xc_l).to(device),
                    torch.from_numpy(live_bundle.station_indices.astype(np.int64)).to(device),
                    torch.from_numpy(anc_l).to(device),
                ).cpu().numpy().astype(np.float32)
            p_lgb = predict_lgbm_grid(lgb_m, xd_l, mk_l, xc_l, live_bundle.station_indices, anc_l)
            p_stk = apply_blend_weights(p_nn, p_lgb, bw)

            st_name_map = stations_active.set_index("station_id")["name"].to_dict()
            for i, sid in enumerate(live_bundle.station_ids):
                days_out = []
                for d_i, dt_s in enumerate(fcast_dates):
                    days_out.append(
                        {
                            "day": d_i + 1,
                            "date": dt_s,
                            "tmin_f": round(float(p_stk[i, d_i, 0]), 1),
                            "tmax_f": round(float(p_stk[i, d_i, 1]), 1),
                            "nn_tmin_f": round(float(p_nn[i, d_i, 0]), 1),
                            "lgbm_tmin_f": round(float(p_lgb[i, d_i, 0]), 1),
                        }
                    )
                live_records.append(
                    {
                        "station_id": str(sid),
                        "network": str(live_bundle.networks[i]),
                        "name": st_name_map.get(str(sid), str(sid)),
                        "forecasts": days_out,
                    }
                )
            print(f"  [{_ts()}] Generated live 5-day forecasts for {len(live_records)} FGZ stations!", flush=True)
    except Exception as exc:
        print(f"  [{_ts()}] Live forecast step skipped ({type(exc).__name__}: {exc})", flush=True)

    master_json = write_single_file_master_bundle(
        runs_dir=runs_dir,
        source_mode="operational_v3_suite",
        coop_audit_df=audit_df,
        channel_coverage_df=cov_df,
        live_forecasts=live_records,
    )
    print(f"\n[{_ts()}] >>> ALL DONE in {time.monotonic() - run_t0:.1f}s | Single Drop-In File: {master_json} <<<", flush=True)


if __name__ == "__main__":
    main()