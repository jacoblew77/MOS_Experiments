#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from fgz_nn_lab.data_ingest import (
    fetch_hourly_predictors_for_stations,
    fetch_iem_coop_observations,
    load_station_catalog,
)
from fgz_nn_lab.evaluate import compute_metrics_bundle, log_experiment_run
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
        if "core_synoptic" in active_groups and any(
            k in name for k in ("t2m_", "td2m_", "dp_spread_", "cloud_", "precip_", "t850_", "t700_", "lapse_")
        ) and not name.startswith("hrrr_"):
            keep = True
        if "pbl_rad_wind" in active_groups and any(
            k in name for k in ("wind_", "gust_", "pblh_", "sw_rad_")
        ) and not name.startswith("hrrr_"):
            keep = True
        if "hrrr_meso" in active_groups and name.startswith("hrrr_"):
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


def evaluate_model_across_folds(
    bundle: TensorBundle,
    x_dyn: np.ndarray,
    mask_dyn: np.ndarray,
    folds: list,
    cfg: dict,
    mode: str,
) -> tuple[dict[str, float], dict[str, float], object]:
    val_preds_all = []
    val_true_all = []
    val_sids_all = []
    val_tiers_all = []
    val_snow_all = []

    test_idx = folds[-1].test_idx
    test_preds_ensemble = np.zeros((len(test_idx), 5, 2), dtype=np.float32)

    for f_i, fold in enumerate(folds, start=1):
        f_label = f"{fold.fold_name} ({f_i}/{len(folds)})"
        tr, va, te = fold.train_idx, fold.val_idx, fold.test_idx
        print(
            f"    [{_ts()}] [{mode.upper()}] Starting {f_label} "
            f"(train={len(tr):,}, val={len(va):,}, test={len(te):,})...",
            flush=True,
        )
        t_fold = time.monotonic()

        scaler = FoldScaler().fit(
            x_dyn[tr],
            mask_dyn[tr],
            bundle.x_ctx[tr],
            bundle.raw_anchor[tr],
            bundle.y_true[tr],
            bundle.station_ids[tr],
        )

        xd_tr, xc_tr = scaler.transform_inputs(x_dyn[tr], mask_dyn[tr], bundle.x_ctx[tr])
        xd_va, xc_va = scaler.transform_inputs(x_dyn[va], mask_dyn[va], bundle.x_ctx[va])
        xd_te, xc_te = scaler.transform_inputs(x_dyn[te], mask_dyn[te], bundle.x_ctx[te])

        anc_tr = scaler.calibrate_anchor(bundle.raw_anchor[tr], bundle.station_ids[tr])
        anc_va = scaler.calibrate_anchor(bundle.raw_anchor[va], bundle.station_ids[va])
        anc_te = scaler.calibrate_anchor(bundle.raw_anchor[te], bundle.station_ids[te])

        if mode == "anchor":
            vp, tp = anc_va, anc_te
        elif mode == "ridge":
            vp = fit_predict_ridge_baseline(
                xd_tr, mask_dyn[tr], xc_tr, anc_tr, bundle.y_true[tr],
                xd_va, mask_dyn[va], xc_va, anc_va,
            )
            tp = fit_predict_ridge_baseline(
                xd_tr, mask_dyn[tr], xc_tr, anc_tr, bundle.y_true[tr],
                xd_te, mask_dyn[te], xc_te, anc_te,
            )
        elif mode == "lgbm":
            vp, tp = fit_predict_lgbm_baseline(
                xd_tr, mask_dyn[tr], xc_tr, anc_tr, bundle.y_true[tr],
                xd_va, mask_dyn[va], xc_va, anc_va, bundle.y_true[va],
                xd_te, mask_dyn[te], xc_te, anc_te,
                seed=int(cfg["seed"]),
                fold_label=fold.fold_name,
            )
        elif mode == "nn":
            vp, tp, _ = train_and_predict_nn(
                xd_tr, mask_dyn[tr], xc_tr, anc_tr, bundle.y_true[tr],
                xd_va, mask_dyn[va], xc_va, anc_va, bundle.y_true[va],
                xd_te, mask_dyn[te], xc_te, anc_te,
                nn_cfg=cfg["nn"],
                seed=int(cfg["seed"]),
                fold_label=fold.fold_name,
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
    return val_summary, test_summary, test_strat


def main() -> None:
    run_t0 = time.monotonic()
    parser = argparse.ArgumentParser(description="FGZ COOP 5-Day Extremum NN Pipeline")
    parser.add_argument(
        "--stage",
        choices=["ingest", "baselines", "train-nn", "ablations", "all"],
        default="all",
    )
    parser.add_argument(
        "--source-mode",
        choices=["era5", "nwp_hybrid"],
        default="era5",
    )
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--stations", type=Path, default=Path("configs/stations_fgz_coop.json"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stations_df = load_station_catalog(args.stations)
    if args.fast_smoke:
        stations_df = stations_df.iloc[:3].reset_index(drop=True)
        start_date = cfg["dates"]["smoke_start"]
        end_date = cfg["dates"]["smoke_end"]
        min_days = 180
        cfg["nn"]["max_epochs"] = 12
    elif args.source_mode == "nwp_hybrid":
        start_date = cfg["dates"]["nwp_start"]
        end_date = cfg["dates"]["nwp_end"]
        min_days = 500
    else:
        start_date = cfg["dates"]["era5_start"]
        end_date = cfg["dates"]["era5_end"]
        min_days = int(cfg["qc_and_gating"]["min_clean_obs_days"])

    raw_dir = Path("data/raw")
    runs_dir = Path("data/runs")

    print(
        f"\n[{_ts()}] >>> [STAGE 1/4: INGEST] mode={args.source_mode}, stage={args.stage}, "
        f"window={start_date}..{end_date}, stations={len(stations_df)} <<<",
        flush=True,
    )
    coop_df = fetch_iem_coop_observations(stations_df, start_date, end_date, raw_dir, min_clean_days=min_days)
    stations_active = stations_df[stations_df["station_id"].isin(coop_df["station_id"].unique())].reset_index(drop=True)
    hourly_by_src = fetch_hourly_predictors_for_stations(
        stations_active, start_date, end_date, raw_dir, source_mode=args.source_mode
    )

    if args.stage == "ingest":
        print(
            f"\n[{_ts()}] >>> INGEST COMPLETE in {time.monotonic() - run_t0:.1f}s | "
            f"Active QC stations: {len(stations_active)}, COOP rows: {len(coop_df):,} <<<",
            flush=True,
        )
        return

    print(
        f"\n[{_ts()}] >>> [STAGE 2/4: ALIGNMENT] Building sub-daily windows, causal lags, and 5-day tensors <<<",
        flush=True,
    )
    qg = cfg["qc_and_gating"]
    bundle = build_aligned_tensors(
        stations_df=stations_active,
        coop_df=coop_df,
        hourly_by_source=hourly_by_src,
        issue_hour_utc=int(qg["issue_hour_utc"]),
        morning_min_gate_utc=int(qg["morning_min_gate_utc"]),
        afternoon_max_gate_utc=int(qg["afternoon_max_gate_utc"]),
        min_causal_lag_hours=int(qg["min_causal_lag_hours"]),
    )
    print(
        f"  [{_ts()}] Summary: {len(bundle.issue_dates):,} 5-day sequences | "
        f"{len(np.unique(bundle.station_ids))} stations | {len(bundle.dyn_feature_names)} dynamic channels",
        flush=True,
    )

    sp = cfg["splits"]
    folds = make_purged_expanding_splits(
        bundle.issue_dates,
        test_days=120 if args.fast_smoke else int(sp["test_days"]),
        val_folds=1 if args.fast_smoke else int(sp["val_folds"]),
        val_fold_days=120 if args.fast_smoke else int(sp["val_fold_days"]),
        purge_days=int(sp["purge_days"]),
    )
    for f in folds:
        tr_s, tr_e = bundle.issue_dates[f.train_idx[0]], bundle.issue_dates[f.train_idx[-1]]
        va_s, va_e = bundle.issue_dates[f.val_idx[0]], bundle.issue_dates[f.val_idx[-1]]
        te_s, te_e = bundle.issue_dates[f.test_idx[0]], bundle.issue_dates[f.test_idx[-1]]
        print(
            f"  [{_ts()}] Split {f.fold_name}: Train[{tr_s}..{tr_e}] | "
            f"Val[{va_s}..{va_e}] | Test[{te_s}..{te_e}]",
            flush=True,
        )

    if args.stage in ("baselines", "all"):
        print(f"\n[{_ts()}] >>> [STAGE 3/4: BASELINES] Running Anchor, Ridge MOS, and LightGBM <<<", flush=True)
        for b_mode in ("anchor", "ridge", "lgbm"):
            v_sum, t_sum, t_strat = evaluate_model_across_folds(
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
            )
            print(
                f"  [{_ts()}] RESULT {b_mode.upper():8s} | "
                f"Test Min MAE (D1-5): {t_sum['mae_min_1_5']:.3f}F | "
                f"Test Max MAE (D1-5): {t_sum['mae_max_1_5']:.3f}F | "
                f"Combined: {t_sum['mae_all_1_5']:.3f}F",
                flush=True,
            )

    if args.stage in ("train-nn", "all"):
        print(f"\n[{_ts()}] >>> [STAGE 4/4: NEURAL NET] Training FiLM Multi-Horizon ResNet <<<", flush=True)
        v_sum, t_sum, t_strat = evaluate_model_across_folds(
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
        )
        print(
            f"  [{_ts()}] RESULT NN_FULL  | "
            f"Test Min MAE (D1-5): {t_sum['mae_min_1_5']:.3f}F | "
            f"Test Max MAE (D1-5): {t_sum['mae_max_1_5']:.3f}F | "
            f"Combined: {t_sum['mae_all_1_5']:.3f}F",
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
            v_sum, t_sum, t_strat = evaluate_model_across_folds(
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
            )
            print(
                f"  [{_ts()}] ABLATION {step_tag:28s} | "
                f"Test Min MAE: {t_sum['mae_min_1_5']:.3f}F | "
                f"Test Max MAE: {t_sum['mae_max_1_5']:.3f}F",
                flush=True,
            )

    print(
        f"\n[{_ts()}] >>> ALL DONE in {time.monotonic() - run_t0:.1f}s | "
        f"Leaderboard: {runs_dir / 'leaderboard.csv'} <<<",
        flush=True,
    )


if __name__ == "__main__":
    main()
