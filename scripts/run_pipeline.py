#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fgz_nn_lab.data_ingest import (
    fetch_hourly_predictors_for_stations,
    fetch_iem_coop_observations,
    load_station_catalog,
)
from fgz_nn_lab.evaluate import (
    compute_metrics_bundle,
    log_experiment_run,
    write_single_file_master_bundle,
)
from fgz_nn_lab.features import TensorBundle, build_aligned_tensors
from fgz_nn_lab.models import (
    fit_predict_lgbm_baseline,
    fit_predict_ridge_baseline,
    train_and_predict_nn,
)
from fgz_nn_lab.splits import FoldScaler, make_purged_expanding_splits


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def apply_ablation_channel_mask(
    bundle: TensorBundle,
    active_groups: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_out = bundle.x_dyn.copy()
    m_out = bundle.mask_dyn.copy()

    for idx, name in enumerate(bundle.dyn_feature_names):
        keep = False
        is_hrrr = name.startswith("hrrr_")
        if "core_synoptic" in active_groups and not is_hrrr and any(
            k in name for k in ("t2m_", "td2m_", "dp_spread_", "cloud_", "precip_", "t850_", "t700_", "lapse_")
        ):
            keep = True
        if "pbl_rad_wind" in active_groups and not is_hrrr and any(
            k in name for k in ("wind_", "gust_", "pblh_", "sw_rad_", "decouple_")
        ):
            keep = True
        if "hrrr_meso" in active_groups and is_hrrr:
            keep = True
        if "snow_soil" in active_groups and any(k in name for k in ("snow_", "soil_")):
            keep = True
        if "ens_spread" in active_groups and name.startswith("ens_spread_"):
            keep = True
        if "causal_lags" in active_groups and name.startswith("causal_"):
            keep = True

        if not keep:
            x_out[:, :, idx] = 0.0
            m_out[:, :, idx] = 0.0

    return x_out, m_out


def pretrain_era5_backbone(
    stations_all: pd.DataFrame,
    raw_dir: Path,
    cfg: dict[str, Any],
    target_cutoff_date: np.datetime64,
    fast_smoke: bool = False,
) -> dict[str, Any]:
    """
    Trains FGZMultiHorizonResNet on the 10-year ERA5 archive strictly BEFORE target_cutoff_date
    so zero future validation/test targets ever leak into NWP fine-tuning.
    """
    e_start = cfg["dates"]["smoke_start"] if fast_smoke else cfg["dates"]["era5_start"]
    e_end = cfg["dates"]["smoke_end"] if fast_smoke else cfg["dates"]["era5_end"]
    min_days = 180 if fast_smoke else int(cfg["qc_and_gating"]["min_clean_obs_days_era5"])

    print(
        f"\n  [{_ts()}] [PRETRAIN] Loading ERA5 archive ({e_start}..{e_end}) with strict leak cutoff < {target_cutoff_date}...",
        flush=True,
    )
    coop_e, _ = fetch_iem_coop_observations(stations_all, e_start, e_end, raw_dir, min_clean_days=min_days)
    st_active_e = stations_all[stations_all["station_id"].isin(coop_e["station_id"].unique())].reset_index(drop=True)
    hourly_e = fetch_hourly_predictors_for_stations(st_active_e, e_start, e_end, raw_dir, source_mode="era5")

    qg = cfg["qc_and_gating"]
    vocab = sorted(stations_all["station_id"].tolist())
    ebundle = build_aligned_tensors(
        stations_df=st_active_e,
        coop_df=coop_e,
        hourly_by_source=hourly_e,
        issue_hour_utc=int(qg["issue_hour_utc"]),
        morning_min_gate_utc=int(qg["morning_min_gate_utc"]),
        afternoon_max_gate_utc=int(qg["afternoon_max_gate_utc"]),
        min_causal_lag_hours=int(qg["min_causal_lag_hours"]),
        station_vocab=vocab,
    )

    # Purge any ERA5 issue_date whose 5-day horizon could touch or cross target_cutoff_date
    safe_cutoff = target_cutoff_date - np.timedelta64(int(cfg["splits"]["purge_days"]), "D")
    elig = np.where(ebundle.issue_dates < safe_cutoff)[0]
    if len(elig) < 200:
        print(f"  [{_ts()}] [PRETRAIN] Too few pre-cutoff ERA5 samples ({len(elig)}); skipping pretrain.", flush=True)
        return {}

    split_pt = int(len(elig) * 0.88)
    tr = elig[:split_pt]
    va = elig[split_pt:]

    sp_cfg = cfg["splits"]
    scaler = FoldScaler(
        min_valid_samples=int(sp_cfg.get("min_valid_channel_samples", 150)),
        clip_zscore=float(sp_cfg.get("clip_zscore", 8.0)),
    ).fit(
        ebundle.x_dyn[tr],
        ebundle.mask_dyn[tr],
        ebundle.x_ctx[tr],
        ebundle.raw_anchor[tr],
        ebundle.y_true[tr],
        ebundle.station_ids[tr],
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
        override_epochs=10 if fast_smoke else int(cfg["nn"].get("pretrain_epochs", 30)),
    )
    return {k: v.cpu() for k, v in pre_model.state_dict().items()}


def evaluate_model_across_folds(
    bundle: TensorBundle,
    x_dyn: np.ndarray,
    mask_dyn: np.ndarray,
    folds: list,
    cfg: dict[str, Any],
    mode: str,
    pretrained_states_by_fold: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame, dict[str, Any]]:
    val_preds_all = []
    val_true_all = []
    val_sids_all = []
    val_tiers_all = []
    val_snow_all = []

    test_idx = folds[-1].test_idx
    test_preds_ensemble = np.zeros((len(test_idx), 5, 2), dtype=np.float32)
    sp_cfg = cfg["splits"]
    diag_out: dict[str, Any] = {}

    for f_i, fold in enumerate(folds, start=1):
        f_label = f"{fold.fold_name} ({f_i}/{len(folds)})"
        tr, va, te = fold.train_idx, fold.val_idx, fold.test_idx
        print(
            f"    [{_ts()}] [{mode.upper()}] Starting {f_label} "
            f"(train={len(tr):,}, val={len(va):,}, test={len(te):,})...",
            flush=True,
        )
        t_fold = time.monotonic()

        scaler = FoldScaler(
            min_valid_samples=30 if len(tr) < 500 else int(sp_cfg.get("min_valid_channel_samples", 150)),
            clip_zscore=float(sp_cfg.get("clip_zscore", 8.0)),
        ).fit(
            x_dyn[tr],
            mask_dyn[tr],
            bundle.x_ctx[tr],
            bundle.raw_anchor[tr],
            bundle.y_true[tr],
            bundle.station_ids[tr],
            bundle.raw_anchor_candidates[tr],
        )

        xd_tr, mk_tr, xc_tr = scaler.transform_inputs(x_dyn[tr], mask_dyn[tr], bundle.x_ctx[tr])
        xd_va, mk_va, xc_va = scaler.transform_inputs(x_dyn[va], mask_dyn[va], bundle.x_ctx[va])
        xd_te, mk_te, xc_te = scaler.transform_inputs(x_dyn[te], mask_dyn[te], bundle.x_ctx[te])

        anc_tr = scaler.calibrate_anchor(bundle.raw_anchor[tr], bundle.station_ids[tr], bundle.raw_anchor_candidates[tr])
        anc_va = scaler.calibrate_anchor(bundle.raw_anchor[va], bundle.station_ids[va], bundle.raw_anchor_candidates[va])
        anc_te = scaler.calibrate_anchor(bundle.raw_anchor[te], bundle.station_ids[te], bundle.raw_anchor_candidates[te])

        active_ch = int(scaler.active_channel_mask.sum()) if scaler.active_channel_mask is not None else 0
        coop24_stations = [s for s, m_id in scaler.station_max_mode.items() if m_id == 2]
        diag_out[fold.fold_name] = {
            "active_channels": active_ch,
            "total_channels": int(x_dyn.shape[2]),
            "coop24_morning_reset_stations": coop24_stations,
        }
        print(
            f"      [{_ts()}] [Scaler {fold.fold_name}] Active non-sparse channels: {active_ch}/{x_dyn.shape[2]} | "
            f"COOP-24h reset stations: {coop24_stations if coop24_stations else 'none'}",
            flush=True,
        )

        if mode == "anchor":
            vp, tp = anc_va, anc_te
        elif mode == "ridge":
            vp = fit_predict_ridge_baseline(
                xd_tr, mk_tr, xc_tr, anc_tr, bundle.y_true[tr],
                xd_va, mk_va, xc_va, anc_va,
            )
            tp = fit_predict_ridge_baseline(
                xd_tr, mk_tr, xc_tr, anc_tr, bundle.y_true[tr],
                xd_te, mk_te, xc_te, anc_te,
            )
        elif mode == "lgbm":
            vp, tp = fit_predict_lgbm_baseline(
                xd_tr, mk_tr, xc_tr, bundle.station_indices[tr], anc_tr, bundle.y_true[tr],
                xd_va, mk_va, xc_va, bundle.station_indices[va], anc_va, bundle.y_true[va],
                xd_te, mk_te, xc_te, bundle.station_indices[te], anc_te,
                seed=int(cfg["seed"]),
                fold_label=fold.fold_name,
            )
        elif mode in ("nn", "nn_pretrained"):
            p_state = (pretrained_states_by_fold or {}).get(fold.fold_name) if mode == "nn_pretrained" else None
            vp, tp, _ = train_and_predict_nn(
                xd_tr, mk_tr, xc_tr, bundle.station_indices[tr], anc_tr, bundle.y_true[tr],
                xd_va, mk_va, xc_va, bundle.station_indices[va], anc_va, bundle.y_true[va],
                xd_te, mk_te, xc_te, bundle.station_indices[te], anc_te,
                nn_cfg=cfg["nn"],
                num_stations=len(bundle.station_vocab),
                seed=int(cfg["seed"]),
                fold_label=f"{fold.fold_name}_{mode}",
                pretrained_state_dict=p_state,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        fold_val_mae = float(np.abs(vp - bundle.y_true[va]).mean())
        print(
            f"    [{_ts()}] [{mode.upper()}] Finished {f_label} in {time.monotonic() - t_fold:.1f}s "
            f"| Fold Val Combined MAE: {fold_val_mae:.3f}F",
            flush=True,
        )

        val_preds_all.append(vp)
        val_true_all.append(bundle.y_true[va])
        val_sids_all.append(bundle.station_ids[va])
        val_tiers_all.append(bundle.tiers[va])
        val_snow_all.append(bundle.snow_flag[va])
        test_preds_ensemble += tp / len(folds)

    val_summary, _ = compute_metrics_bundle(
        np.concatenate(val_preds_all, axis=0),
        np.concatenate(val_true_all, axis=0),
        np.concatenate(val_sids_all, axis=0),
        np.concatenate(val_tiers_all, axis=0),
        np.concatenate(val_snow_all, axis=0),
    )
    test_summary, test_strat = compute_metrics_bundle(
        test_preds_ensemble,
        bundle.y_true[test_idx],
        bundle.station_ids[test_idx],
        bundle.tiers[test_idx],
        bundle.snow_flag[test_idx],
    )
    return val_summary, test_summary, test_strat, diag_out


def main() -> None:
    run_t0 = time.monotonic()
    parser = argparse.ArgumentParser(description="FGZ COOP 5-Day Extremum NN Pipeline v2")
    parser.add_argument(
        "--stage",
        choices=["ingest", "baselines", "train-nn", "ablations", "all"],
        default="all",
    )
    parser.add_argument(
        "--source-mode",
        choices=["era5", "nwp_hybrid", "true_lead_nwp"],
        default="era5",
    )
    parser.add_argument(
        "--pretrain-era5",
        action="store_true",
        help="Also train & evaluate a two-stage ERA5-pretrained -> NWP-finetuned NN",
    )
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--stations", type=Path, default=Path("configs/stations_fgz_coop.json"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stations_all = load_station_catalog(args.stations)
    vocab = sorted(stations_all["station_id"].tolist())

    qg = cfg["qc_and_gating"]
    if args.fast_smoke:
        stations_df = stations_all.iloc[:3].reset_index(drop=True)
        start_date = cfg["dates"]["smoke_start"]
        end_date = cfg["dates"]["smoke_end"]
        min_days = 180
        cfg["nn"]["max_epochs"] = 12
    elif args.source_mode == "true_lead_nwp":
        stations_df = stations_all
        start_date = cfg["dates"]["true_lead_start"]
        end_date = cfg["dates"]["true_lead_end"]
        min_days = int(qg.get("min_clean_obs_days_lead", 300))
    elif args.source_mode == "nwp_hybrid":
        stations_df = stations_all
        start_date = cfg["dates"]["nwp_start"]
        end_date = cfg["dates"]["nwp_end"]
        min_days = int(qg.get("min_clean_obs_days_nwp", 365))
    else:
        stations_df = stations_all
        start_date = cfg["dates"]["era5_start"]
        end_date = cfg["dates"]["era5_end"]
        min_days = int(qg.get("min_clean_obs_days_era5", 730))

    raw_dir = Path("data/raw")
    runs_dir = Path("data/runs")

    print(
        f"\n[{_ts()}] >>> [STAGE 1/4: INGEST] mode={args.source_mode}, stage={args.stage}, "
        f"window={start_date}..{end_date}, stations={len(stations_df)} <<<",
        flush=True,
    )
    coop_df, coop_audit_df = fetch_iem_coop_observations(
        stations_df, start_date, end_date, raw_dir, min_clean_days=min_days
    )
    stations_active = stations_df[stations_df["station_id"].isin(coop_df["station_id"].unique())].reset_index(drop=True)
    hourly_by_src = fetch_hourly_predictors_for_stations(
        stations_active, start_date, end_date, raw_dir, source_mode=args.source_mode
    )

    if args.stage == "ingest":
        print(
            f"\n[{_ts()}] >>> INGEST COMPLETE in {time.monotonic() - run_t0:.1f}s | "
            f"Active QC stations: {len(stations_active)}/{len(stations_df)}, COOP rows: {len(coop_df):,} <<<",
            flush=True,
        )
        return

    print(
        f"\n[{_ts()}] >>> [STAGE 2/4: ALIGNMENT] Building sub-daily windows, COOP-24h windows & 5-day tensors <<<",
        flush=True,
    )
    bundle = build_aligned_tensors(
        stations_df=stations_active,
        coop_df=coop_df,
        hourly_by_source=hourly_by_src,
        issue_hour_utc=int(qg["issue_hour_utc"]),
        morning_min_gate_utc=int(qg["morning_min_gate_utc"]),
        afternoon_max_gate_utc=int(qg["afternoon_max_gate_utc"]),
        min_causal_lag_hours=int(qg["min_causal_lag_hours"]),
        station_vocab=vocab,
    )

    cov_rows = []
    for ch_i, ch_name in enumerate(bundle.dyn_feature_names):
        valid_frac_d1 = float(bundle.mask_dyn[:, 0, ch_i].mean())
        valid_frac_d5 = float(bundle.mask_dyn[:, 4, ch_i].mean())
        cov_rows.append(
            {
                "channel": ch_name,
                "valid_frac_day1": round(valid_frac_d1, 4),
                "valid_frac_day5": round(valid_frac_d5, 4),
            }
        )
    channel_cov_df = pd.DataFrame(cov_rows)

    sp = cfg["splits"]
    folds = make_purged_expanding_splits(
        bundle.issue_dates,
        test_days=120 if args.fast_smoke else int(sp["test_days"]),
        val_folds=1 if args.fast_smoke else int(sp["val_folds"]),
        val_fold_days=120 if args.fast_smoke else int(sp["val_fold_days"]),
        purge_days=int(sp["purge_days"]),
    )
    for f in folds:
        print(
            f"  [{_ts()}] Split {f.fold_name}: Train[{bundle.issue_dates[f.train_idx[0]]}..{bundle.issue_dates[f.train_idx[-1]]}] | "
            f"Val[{bundle.issue_dates[f.val_idx[0]]}..{bundle.issue_dates[f.val_idx[-1]]}] | "
            f"Test[{bundle.issue_dates[f.test_idx[0]]}..{bundle.issue_dates[f.test_idx[-1]]}]",
            flush=True,
        )

    if args.stage in ("baselines", "all"):
        print(f"\n[{_ts()}] >>> [STAGE 3/4: BASELINES] Running Anchor, Ridge MOS, and LightGBM <<<", flush=True)
        for b_mode in ("anchor", "ridge", "lgbm"):
            v_sum, t_sum, t_strat, f_diag = evaluate_model_across_folds(
                bundle, bundle.x_dyn, bundle.mask_dyn, folds, cfg, mode=b_mode
            )
            log_experiment_run(
                runs_dir,
                run_id=f"{args.source_mode}_{b_mode}_full",
                model_family=b_mode,
                feature_tag="all_features",
                val_summary=v_sum,
                test_summary=t_sum,
                stratified_df=t_strat,
                config_snapshot=cfg,
                fold_diagnostics=f_diag,
            )
            print(
                f"  [{_ts()}] RESULT {b_mode.upper():8s} | "
                f"Test Min MAE (D1-5): {t_sum['mae_min_1_5']:.3f}F | "
                f"Test Max MAE (D1-5): {t_sum['mae_max_1_5']:.3f}F | "
                f"Combined: {t_sum['mae_all_1_5']:.3f}F",
                flush=True,
            )

    if args.stage in ("train-nn", "all"):
        print(f"\n[{_ts()}] >>> [STAGE 4/4: NEURAL NET] Training FiLM + Station-Embed ResNet <<<", flush=True)
        v_sum, t_sum, t_strat, f_diag = evaluate_model_across_folds(
            bundle, bundle.x_dyn, bundle.mask_dyn, folds, cfg, mode="nn"
        )
        log_experiment_run(
            runs_dir,
            run_id=f"{args.source_mode}_nn_res_film_full",
            model_family="nn_res_film",
            feature_tag="all_features",
            val_summary=v_sum,
            test_summary=t_sum,
            stratified_df=t_strat,
            config_snapshot=cfg,
            fold_diagnostics=f_diag,
        )
        print(
            f"  [{_ts()}] RESULT NN_FULL  | "
            f"Test Min MAE (D1-5): {t_sum['mae_min_1_5']:.3f}F | "
            f"Test Max MAE (D1-5): {t_sum['mae_max_1_5']:.3f}F | "
            f"Combined: {t_sum['mae_all_1_5']:.3f}F",
            flush=True,
        )

        if args.pretrain_era5 and args.source_mode != "era5":
            print(f"\n[{_ts()}] >>> [STAGE 4B: ERA5 -> NWP TRANSFER] Pretraining per fold with strict leak cutoff <<<", flush=True)
            pre_states: dict[str, dict[str, Any]] = {}
            for f in folds:
                val_start_dt = bundle.issue_dates[f.val_idx[0]]
                pre_states[f.fold_name] = pretrain_era5_backbone(
                    stations_all=stations_all,
                    raw_dir=raw_dir,
                    cfg=cfg,
                    target_cutoff_date=val_start_dt,
                    fast_smoke=args.fast_smoke,
                )
            v_sum_p, t_sum_p, t_strat_p, f_diag_p = evaluate_model_across_folds(
                bundle,
                bundle.x_dyn,
                bundle.mask_dyn,
                folds,
                cfg,
                mode="nn_pretrained",
                pretrained_states_by_fold=pre_states,
            )
            log_experiment_run(
                runs_dir,
                run_id=f"{args.source_mode}_nn_era5_pretrained",
                model_family="nn_era5_pretrained",
                feature_tag="era5_transfer_all_features",
                val_summary=v_sum_p,
                test_summary=t_sum_p,
                stratified_df=t_strat_p,
                config_snapshot=cfg,
                fold_diagnostics=f_diag_p,
            )
            print(
                f"  [{_ts()}] RESULT NN_XFER  | "
                f"Test Min MAE (D1-5): {t_sum_p['mae_min_1_5']:.3f}F | "
                f"Test Max MAE (D1-5): {t_sum_p['mae_max_1_5']:.3f}F | "
                f"Combined: {t_sum_p['mae_all_1_5']:.3f}F",
                flush=True,
            )

    if args.stage == "ablations":
        print(f"\n[{_ts()}] >>> [ABLATIONS] Running 6-step physical predictor ladder <<<", flush=True)
        ladder = [
            ("step_a_site_season_only", set()),
            ("step_b_add_core_synoptic", {"core_synoptic"}),
            ("step_c_add_pbl_rad_wind", {"core_synoptic", "pbl_rad_wind"}),
            ("step_d_add_hrrr_meso", {"core_synoptic", "pbl_rad_wind", "hrrr_meso"}),
            ("step_e_add_snow_soil", {"core_synoptic", "pbl_rad_wind", "hrrr_meso", "snow_soil"}),
            ("step_f_add_spread_and_lags", {"core_synoptic", "pbl_rad_wind", "hrrr_meso", "snow_soil", "ens_spread", "causal_lags"}),
        ]
        for s_idx, (step_tag, groups) in enumerate(ladder, start=1):
            print(f"\n  [{_ts()}] --- Ablation [{s_idx}/{len(ladder)}]: {step_tag} ---", flush=True)
            x_abl, m_abl = apply_ablation_channel_mask(bundle, groups)
            v_sum, t_sum, t_strat, f_diag = evaluate_model_across_folds(
                bundle, x_abl, m_abl, folds, cfg, mode="nn"
            )
            log_experiment_run(
                runs_dir,
                run_id=f"{args.source_mode}_abl_{step_tag}",
                model_family="nn_res_film",
                feature_tag=step_tag,
                val_summary=v_sum,
                test_summary=t_sum,
                stratified_df=t_strat,
                config_snapshot=cfg,
                fold_diagnostics=f_diag,
            )
            print(
                f"  [{_ts()}] ABLATION {step_tag:28s} | "
                f"Test Min MAE: {t_sum['mae_min_1_5']:.3f}F | "
                f"Test Max MAE: {t_sum['mae_max_1_5']:.3f}F | "
                f"Combined: {t_sum['mae_all_1_5']:.3f}F",
                flush=True,
            )

    master_json, master_csv = write_single_file_master_bundle(
        runs_dir=runs_dir,
        source_mode=args.source_mode,
        coop_audit_df=coop_audit_df,
        channel_coverage_df=channel_cov_df,
    )
    print(
        f"\n[{_ts()}] >>> ALL DONE in {time.monotonic() - run_t0:.1f}s <<<\n"
        f"  Single-file JSON drop-in : {master_json}\n"
        f"  Single-file CSV drop-in  : {master_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()