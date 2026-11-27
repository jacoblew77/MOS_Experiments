#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from fgz_nn_lab.dashboard import build_github_pages_site
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
    CANONICAL_DYN_FEATURES,
    build_aligned_tensors,
    build_live_inference_bundle,
)
from fgz_nn_lab.models import (
    FGZMultiHorizonResNet,
    apply_network_aware_blend,
    fit_conformal_residual_quantiles,
    fit_network_aware_blend_weights,
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
    e_start = cfg["dates"]["smoke_start"] if fast_smoke else "2021-01-01"
    e_end = cfg["dates"]["smoke_end"] if fast_smoke else cfg["dates"]["era5_end"]

    print(f"    [{_ts()}] [PRETRAIN] Loading 100%-real obs + ERA5 ({e_start}..{e_end}, cutoff < {target_cutoff_date})...", flush=True)
    obs_e, _, _ = fetch_unestimated_iem_observations(
        stations_active, e_start, e_end, raw_dir,
        min_clean_days=120 if fast_smoke else 350,
        min_recent_days=50,
        max_coop=30, max_asos=30,
        check_estimated_txt=False,
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


def generate_live_forecasts_from_checkpoint(
    ckpt: dict[str, Any],
    raw_dir: Path,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    stations_active: pd.DataFrame = ckpt["stations_active"]
    vocab: list[str] = ckpt["vocab"]
    sc: FoldScaler = ckpt["scaler"]
    lgb_m = ckpt["lgb_models"]
    weight_map: dict[str, np.ndarray] = ckpt["weight_map"]
    quantiles_map: dict[str, dict[str, np.ndarray]] = ckpt["quantiles_map"]
    nn_cfg = ckpt["nn_cfg"]

    # Fetch only the last 14 days of real IEM daily.py obs for causal lags (0 Open-Meteo credits!)
    today_utc = pd.Timestamp(datetime.now(timezone.utc).date())
    rec_start = (today_utc - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    rec_end = today_utc.strftime("%Y-%m-%d")
    recent_obs, _, _ = fetch_unestimated_iem_observations(
        stations_active, rec_start, rec_end, raw_dir / "live_obs_tmp",
        min_clean_days=0, min_recent_days=0, max_coop=50, max_asos=50, check_estimated_txt=False,
    )

    live_hourly, live_mos = fetch_live_operational_inputs(stations_active)
    qg = cfg["qc_and_gating"]
    live_bundle, fcast_dates = build_live_inference_bundle(
        stations_active, recent_obs, live_hourly, live_mos, vocab,
        int(qg["issue_hour_utc"]), int(qg["morning_min_gate_utc"]),
        int(qg["afternoon_max_gate_utc"]), int(qg["min_causal_lag_hours"]),
    )
    if live_bundle is None:
        return []

    xd_l, mk_l, xc_l = sc.transform_inputs(live_bundle.x_dyn, live_bundle.mask_dyn, live_bundle.x_ctx)
    anc_l = sc.calibrate_anchor(live_bundle.raw_anchor, live_bundle.station_ids, live_bundle.raw_anchor_candidates)

    nn_model = FGZMultiHorizonResNet(
        dyn_dim=len(CANONICAL_DYN_FEATURES),
        ctx_dim=live_bundle.x_ctx.shape[1],
        num_stations=len(vocab),
        hidden_dim=int(nn_cfg.get("hidden_dim", 96)),
        ctx_embed_dim=int(nn_cfg.get("ctx_embed_dim", 32)),
        station_embed_dim=int(nn_cfg.get("station_embed_dim", 12)),
        dropout=float(nn_cfg.get("dropout", 0.18)),
        channel_dropout=float(nn_cfg.get("channel_dropout", 0.12)),
    )
    nn_model.load_state_dict(ckpt["nn_state_dict"], strict=True)
    nn_model.eval()

    with torch.no_grad():
        p_nn = nn_model(
            torch.from_numpy(xd_l),
            torch.from_numpy(mk_l),
            torch.from_numpy(xc_l),
            torch.from_numpy(live_bundle.station_indices.astype(np.int64)),
            torch.from_numpy(anc_l),
        ).numpy().astype(np.float32)

    p_lgb = predict_lgbm_grid(lgb_m, xd_l, mk_l, xc_l, live_bundle.station_indices, anc_l)
    p_stk = apply_network_aware_blend(p_nn, p_lgb, live_bundle.networks, weight_map)
    # Enforce physical non-crossing (Min <= Max)
    p_stk[:, :, 0] = np.minimum(p_stk[:, :, 0], p_stk[:, :, 1] - 0.5)

    mode_min_lbl = {0: "cal", 1: "morn24", 2: "eve24"}
    mode_max_lbl = {3: "cal", 4: "morn24", 5: "eve24"}
    st_meta = stations_active.set_index("station_id").to_dict(orient="index")

    live_records: list[dict[str, Any]] = []
    for i, sid in enumerate(live_bundle.station_ids):
        s_str = str(sid)
        net_s = str(live_bundle.networks[i])
        meta = st_meta.get(s_str, {})
        q_net = quantiles_map.get(net_s, quantiles_map["GLOBAL"])
        q10, q90 = q_net["q10"], q_net["q90"]
        clock_str = f"{mode_min_lbl.get(sc.station_min_mode.get(s_str, 0))}/{mode_max_lbl.get(sc.station_max_mode.get(s_str, 3))}"

        days_out = []
        for d_i, dt_s in enumerate(fcast_dates):
            lo = float(p_stk[i, d_i, 0])
            hi = float(p_stk[i, d_i, 1])
            days_out.append(
                {
                    "day": d_i + 1,
                    "date": dt_s,
                    "tmin_f": round(lo, 1),
                    "tmin_p10_f": round(lo + float(q10[d_i, 0]), 1),
                    "tmin_p90_f": round(lo + float(q90[d_i, 0]), 1),
                    "tmax_f": round(hi, 1),
                    "tmax_p10_f": round(hi + float(q10[d_i, 1]), 1),
                    "tmax_p90_f": round(hi + float(q90[d_i, 1]), 1),
                    "nn_tmin_f": round(float(p_nn[i, d_i, 0]), 1),
                    "lgbm_tmin_f": round(float(p_lgb[i, d_i, 0]), 1),
                    "anchor_tmin_f": round(float(anc_l[i, d_i, 0]), 1),
                    "nn_tmax_f": round(float(p_nn[i, d_i, 1]), 1),
                    "lgbm_tmax_f": round(float(p_lgb[i, d_i, 1]), 1),
                    "anchor_tmax_f": round(float(anc_l[i, d_i, 1]), 1),
                }
            )
        live_records.append(
            {
                "station_id": s_str,
                "network": net_s,
                "name": str(meta.get("name", s_str)),
                "lat": float(meta.get("lat", 35.14)),
                "lon": float(meta.get("lon", -111.67)),
                "elev_ft": float(meta.get("elev_ft", 7000.0)),
                "tier": str(live_bundle.tiers[i]),
                "obs_clock": clock_str,
                "forecasts": days_out,
            }
        )
    return live_records


def main() -> None:
    run_t0 = time.monotonic()
    parser = argparse.ArgumentParser(description="WFO Flagstaff Operational AI-MOS v4 (Train + Fast Daily Forecast + GitHub Pages)")
    parser.add_argument(
        "--mode",
        choices=["auto", "forecast", "retrain"],
        default="auto",
        help="'auto' uses cached checkpoint (<14d old) for ~40s forecast or retrains if absent; 'forecast' forces fast forecast if checkpoint exists; 'retrain' forces full benchmark training.",
    )
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--stations", type=Path, default=Path("configs/stations_fgz_coop.json"))
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    st_cfg = yaml.safe_load(args.stations.read_text(encoding="utf-8"))
    raw_dir = Path("data/raw")
    runs_dir = Path("data/runs")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "fgz_op_v4_checkpoint.pkl"

    # Check if we can execute Fast 40-Second Daily Forecast Mode (~106 API credits!)
    use_fast_ckpt = False
    if ckpt_path.exists() and args.mode in ("auto", "forecast") and not args.fast_smoke:
        ckpt_obj = pickle.loads(ckpt_path.read_bytes())
        trained_dt = pd.Timestamp(ckpt_obj.get("trained_at_utc", "2000-01-01T00:00:00Z"))
        age_days = (pd.Timestamp(datetime.now(timezone.utc)) - trained_dt).total_seconds() / 86400.0
        expected_n = int(st_cfg.get("max_asos_stations", 12)) + int(st_cfg.get("max_coop_stations", 14))
        if args.mode == "forecast" or (age_days <= 14.0 and len(ckpt_obj.get("vocab", [])) == expected_n):
            use_fast_ckpt = True

    if use_fast_ckpt:
        print(f"\n[{_ts()}] >>> [FAST DAILY FORECAST MODE] Loaded cached checkpoint ({ckpt_path.name}) — ~106 API credits! <<<", flush=True)
        ckpt_obj = pickle.loads(ckpt_path.read_bytes())
        live_records = generate_live_forecasts_from_checkpoint(ckpt_obj, raw_dir, cfg)
        site_index = build_github_pages_site(
            site_dir=args.site_dir,
            stations_df=ckpt_obj["stations_active"],
            live_forecasts=live_records,
            leaderboard_rows=ckpt_obj.get("leaderboard_rows", []),
            station_mae_map=ckpt_obj.get("station_mae_map", {}),
            run_mode="fast_daily_checkpoint_v4",
        )
        runs_dir.mkdir(parents=True, exist_ok=True)
        if ckpt_obj.get("leaderboard_rows"):
            pd.DataFrame(ckpt_obj["leaderboard_rows"]).to_csv(runs_dir / "leaderboard.csv", index=False)
        master_json = write_single_file_master_bundle(
            runs_dir=runs_dir,
            source_mode="operational_v4_fast_daily",
            coop_audit_df=ckpt_obj.get("audit_df", pd.DataFrame()),
            channel_coverage_df=ckpt_obj.get("cov_df", pd.DataFrame()),
            live_forecasts=live_records,
        )
        print(
            f"\n[{_ts()}] >>> FAST DAILY FORECAST COMPLETE in {time.monotonic() - run_t0:.1f}s <<<\n"
            f"  GitHub Pages Portal : {site_index}\n"
            f"  Single Master JSON  : {master_json}",
            flush=True,
        )
        return

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
        max_coop=int(st_cfg.get("max_coop_stations", 14)),
        max_asos=int(st_cfg.get("max_asos_stations", 12)),
    )
    vocab = sorted(stations_active["station_id"].tolist())

    print(f"\n[{_ts()}] >>> [2/5] INGESTING INCREMENTAL DELTA NWP, 2.5KM NBM & D1-5 GFS+MEX MOS <<<", flush=True)
    mos_df = fetch_iem_mos_bulletins(stations_active, start_date, end_date, raw_dir)
    hourly_by_src = fetch_hourly_predictors_for_stations(
        stations_active, start_date, end_date, raw_dir, source_mode="true_lead_nwp"
    )

    print(f"\n[{_ts()}] >>> [3/5] ALIGNING 5-DAY MULTI-MODEL & EXTENDED MOS TENSORS <<<", flush=True)
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

    print(f"\n[{_ts()}] >>> [4/5] TRAINING MODELS, NETWORK-AWARE (ASOS/COOP) STACK & CONFORMAL P10-P90 <<<", flush=True)
    test_idx = folds[-1].test_idx
    n_folds = len(folds)

    mod_keys = ("anchor", "ridge", "lgbm", "nn_scratch", "nn_era5_pretrained", "nn_lgbm_net_stack")
    preds_val_by_mod: dict[str, list[np.ndarray]] = {k: [] for k in mod_keys}
    preds_test_ens: dict[str, np.ndarray] = {k: np.zeros((len(test_idx), 5, 2), dtype=np.float32) for k in mod_keys}
    val_y_all, val_sid_all, val_tier_all, val_snow_all, val_net_all = [], [], [], [], []
    fold_diag: dict[str, Any] = {}
    last_ckpt_parts: dict[str, Any] = {}

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

        vp_anc, tp_anc = anc_va, anc_te
        vp_rid = fit_predict_ridge_baseline(xd_tr, mk_tr, xc_tr, anc_tr, bundle.y_true[tr], xd_va, mk_va, xc_va, anc_va)
        tp_rid = fit_predict_ridge_baseline(xd_tr, mk_tr, xc_tr, anc_tr, bundle.y_true[tr], xd_te, mk_te, xc_te, anc_te)

        vp_lgb, tp_lgb, lgb_models = fit_predict_lgbm_baseline(
            xd_tr, mk_tr, xc_tr, bundle.station_indices[tr], anc_tr, bundle.y_true[tr],
            xd_va, mk_va, xc_va, bundle.station_indices[va], anc_va, bundle.y_true[va],
            xd_te, mk_te, xc_te, bundle.station_indices[te], anc_te,
            seed=int(cfg["seed"]), fold_label=fold.fold_name,
        )

        vp_nns, tp_nns, _ = train_and_predict_nn(
            xd_tr, mk_tr, xc_tr, bundle.station_indices[tr], anc_tr, bundle.y_true[tr],
            xd_va, mk_va, xc_va, bundle.station_indices[va], anc_va, bundle.y_true[va],
            xd_te, mk_te, xc_te, bundle.station_indices[te], anc_te,
            nn_cfg=cfg["nn"], num_stations=len(vocab), seed=int(cfg["seed"]),
            fold_label=f"{fold.fold_name}_scratch",
        )

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

        # Network-Aware (AZ_ASOS vs AZ_COOP) L1 Stack + Conformal P10-P90 Quantiles
        net_weights = fit_network_aware_blend_weights(vp_nnx, vp_lgb, bundle.y_true[va], bundle.networks[va])
        vp_stk = apply_network_aware_blend(vp_nnx, vp_lgb, bundle.networks[va], net_weights)
        tp_stk = apply_network_aware_blend(tp_nnx, tp_lgb, bundle.networks[te], net_weights)
        q_map = fit_conformal_residual_quantiles(vp_stk, bundle.y_true[va], bundle.networks[va])

        mode_names_min = {0: "cal", 1: "morn24", 2: "eve24"}
        mode_names_max = {3: "cal", 4: "morn24", 5: "eve24"}
        fold_diag[fold.fold_name] = {
            "active_channels": int(scaler.active_channel_mask.sum()) if scaler.active_channel_mask is not None else 0,
            "total_channels": int(bundle.x_dyn.shape[2]),
            "station_window_modes": {
                s: f"min={mode_names_min.get(scaler.station_min_mode.get(s, 0))},max={mode_names_max.get(scaler.station_max_mode.get(s, 3))}"
                for s in sorted(np.unique(bundle.station_ids[tr]))
            },
            "nn_weight_asos_min_d1_5": [round(float(x), 2) for x in net_weights["AZ_ASOS"][:, 0]],
            "nn_weight_coop_min_d1_5": [round(float(x), 2) for x in net_weights["AZ_COOP"][:, 0]],
            "nn_weight_asos_max_d1_5": [round(float(x), 2) for x in net_weights["AZ_ASOS"][:, 1]],
            "nn_weight_coop_max_d1_5": [round(float(x), 2) for x in net_weights["AZ_COOP"][:, 1]],
        }

        for m_key, vp_arr, tp_arr in [
            ("anchor", vp_anc, tp_anc),
            ("ridge", vp_rid, tp_rid),
            ("lgbm", vp_lgb, tp_lgb),
            ("nn_scratch", vp_nns, tp_nns),
            ("nn_era5_pretrained", vp_nnx, tp_nnx),
            ("nn_lgbm_net_stack", vp_stk, tp_stk),
        ]:
            preds_val_by_mod[m_key].append(vp_arr)
            preds_test_ens[m_key] += tp_arr / n_folds

        val_y_all.append(bundle.y_true[va])
        val_sid_all.append(bundle.station_ids[va])
        val_tier_all.append(bundle.tiers[va])
        val_snow_all.append(bundle.snow_flag[va])
        val_net_all.append(bundle.networks[va])

        last_ckpt_parts = {
            "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stations_active": stations_active,
            "vocab": vocab,
            "scaler": scaler,
            "nn_state_dict": {k: v.cpu() for k, v in nnx_model.state_dict().items()},
            "nn_cfg": cfg["nn"],
            "lgb_models": lgb_models,
            "weight_map": net_weights,
            "quantiles_map": q_map,
            "audit_df": audit_df,
            "cov_df": cov_df,
        }

    vy_cat = np.concatenate(val_y_all, axis=0)
    vsid_cat = np.concatenate(val_sid_all, axis=0)
    vtier_cat = np.concatenate(val_tier_all, axis=0)
    vsnow_cat = np.concatenate(val_snow_all, axis=0)
    vnet_cat = np.concatenate(val_net_all, axis=0)

    station_mae_map: dict[str, dict[str, float]] = {}
    print(f"\n[{_ts()}] >>> OPERATIONAL v4 TEST LEADERBOARD (26 FGZ SITES, 0% ESTIMATED OBS) <<<", flush=True)
    for m_key in mod_keys:
        v_sum, _ = compute_metrics_bundle(
            np.concatenate(preds_val_by_mod[m_key], axis=0), vy_cat, vsid_cat, vtier_cat, vsnow_cat, vnet_cat
        )
        t_sum, t_strat = compute_metrics_bundle(
            preds_test_ens[m_key], bundle.y_true[test_idx], bundle.station_ids[test_idx],
            bundle.tiers[test_idx], bundle.snow_flag[test_idx], bundle.networks[test_idx],
        )
        log_experiment_run(
            runs_dir, run_id=f"op_v4_{m_key}", model_family=m_key,
            feature_tag="nbm_mex_mos_netstack_v4", val_summary=v_sum, test_summary=t_sum,
            stratified_df=t_strat, config_snapshot=cfg, fold_diagnostics=fold_diag,
        )
        if m_key == "nn_lgbm_net_stack":
            for _, sr in t_strat[t_strat["slice_type"] == "station"].iterrows():
                station_mae_map[str(sr["slice_name"])] = {
                    "mae_min": float(sr["mae_min"]),
                    "mae_max": float(sr["mae_max"]),
                }
        print(
            f"  {m_key.upper():22s} | Test Min MAE: {t_sum['mae_min_1_5']:.3f}F | "
            f"Test Max MAE: {t_sum['mae_max_1_5']:.3f}F | Combined: {t_sum['mae_all_1_5']:.3f}F",
            flush=True,
        )

    lb_rows = pd.read_csv(runs_dir / "leaderboard.csv").to_dict(orient="records")
    last_ckpt_parts["leaderboard_rows"] = lb_rows
    last_ckpt_parts["station_mae_map"] = station_mae_map

    if not args.fast_smoke:
        ckpt_path.write_bytes(pickle.dumps(last_ckpt_parts))
        print(f"  [{_ts()}] Saved production checkpoint -> {ckpt_path} (future daily runs take ~40s!)", flush=True)

    print(f"\n[{_ts()}] >>> [5/5] GENERATING LIVE 5-DAY FORECAST & INTERACTIVE GITHUB PAGES PORTAL <<<", flush=True)
    live_records: list[dict[str, Any]] = []
    try:
        live_records = generate_live_forecasts_from_checkpoint(last_ckpt_parts, raw_dir, cfg)
        print(f"  [{_ts()}] Generated live 5-day forecasts + P10-P90 intervals for {len(live_records)} FGZ stations!", flush=True)
    except Exception as exc:
        print(f"  [{_ts()}] Live forecast step skipped ({type(exc).__name__}: {exc})", flush=True)

    site_index = build_github_pages_site(
        site_dir=args.site_dir,
        stations_df=stations_active,
        live_forecasts=live_records,
        leaderboard_rows=lb_rows,
        station_mae_map=station_mae_map,
        run_mode="full_train_and_forecast_v4",
    )
    master_json = write_single_file_master_bundle(
        runs_dir=runs_dir,
        source_mode="operational_v4_suite",
        coop_audit_df=audit_df,
        channel_coverage_df=cov_df,
        live_forecasts=live_records,
    )
    print(
        f"\n[{_ts()}] >>> ALL DONE in {time.monotonic() - run_t0:.1f}s <<<\n"
        f"  GitHub Pages Portal : {site_index}\n"
        f"  Single Master JSON  : {master_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()