from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compute_metrics_bundle(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    station_ids: np.ndarray,
    tiers: np.ndarray,
    snow_flag: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    abs_err = np.abs(y_pred - y_true)
    min_err = abs_err[:, :, 0]
    max_err = abs_err[:, :, 1]

    summary: dict[str, float] = {
        "mae_min_1_5": float(min_err.mean()),
        "mae_max_1_5": float(max_err.mean()),
        "mae_all_1_5": float(abs_err.mean()),
    }
    for d in range(5):
        summary[f"mae_min_d{d + 1}"] = float(min_err[:, d].mean())
        summary[f"mae_max_d{d + 1}"] = float(max_err[:, d].mean())

    slice_rows: list[dict[str, Any]] = []

    freeze_mask = y_true[:, :, 0] <= 32.0
    if freeze_mask.any():
        slice_rows.append(
            {
                "slice_type": "regime",
                "slice_name": "freezing_nights_le_32F",
                "samples": int(freeze_mask.sum()),
                "mae_min": round(float(min_err[freeze_mask].mean()), 4),
                "mae_max": round(float(max_err[freeze_mask].mean()), 4),
            }
        )

    if snow_flag.any():
        slice_rows.append(
            {
                "slice_type": "regime",
                "slice_name": "snow_present",
                "samples": int(snow_flag.sum()),
                "mae_min": round(float(min_err[snow_flag].mean()), 4),
                "mae_max": round(float(max_err[snow_flag].mean()), 4),
            }
        )

    for tier in np.unique(tiers):
        m = tiers == tier
        slice_rows.append(
            {
                "slice_type": "tier",
                "slice_name": str(tier),
                "samples": int(m.sum() * 5),
                "mae_min": round(float(min_err[m].mean()), 4),
                "mae_max": round(float(max_err[m].mean()), 4),
            }
        )

    for sid in np.unique(station_ids):
        m = station_ids == sid
        slice_rows.append(
            {
                "slice_type": "station",
                "slice_name": str(sid),
                "samples": int(m.sum() * 5),
                "mae_min": round(float(min_err[m].mean()), 4),
                "mae_max": round(float(max_err[m].mean()), 4),
            }
        )

    return summary, pd.DataFrame(slice_rows)


def log_experiment_run(
    runs_dir: Path,
    run_id: str,
    model_family: str,
    feature_tag: str,
    val_summary: dict[str, float],
    test_summary: dict[str, float],
    stratified_df: pd.DataFrame,
    config_snapshot: dict[str, Any],
    fold_diagnostics: dict[str, Any] | None = None,
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "model_family": model_family,
        "feature_set_tag": feature_tag,
        "val": {k: round(v, 4) for k, v in val_summary.items()},
        "test": {k: round(v, 4) for k, v in test_summary.items()},
        "fold_diagnostics": fold_diagnostics or {},
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "resolved_config.json").write_text(json.dumps(config_snapshot, indent=2, default=str), encoding="utf-8")
    stratified_df.to_csv(run_dir / "metrics_stratified.csv", index=False)

    leaderboard_path = runs_dir / "leaderboard.csv"
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_family": model_family,
        "feature_set_tag": feature_tag,
        "val_mae_min_1_5": round(val_summary["mae_min_1_5"], 4),
        "val_mae_max_1_5": round(val_summary["mae_max_1_5"], 4),
        "val_mae_all_1_5": round(val_summary["mae_all_1_5"], 4),
        "test_mae_min_1_5": round(test_summary["mae_min_1_5"], 4),
        "test_mae_max_1_5": round(test_summary["mae_max_1_5"], 4),
        "test_mae_all_1_5": round(test_summary["mae_all_1_5"], 4),
    }
    for d in range(1, 6):
        row[f"test_min_d{d}"] = round(test_summary[f"mae_min_d{d}"], 4)
        row[f"test_max_d{d}"] = round(test_summary[f"mae_max_d{d}"], 4)

    row_df = pd.DataFrame([row])
    if leaderboard_path.exists():
        prev = pd.read_csv(leaderboard_path)
        prev = prev[prev["run_id"] != run_id]
        combined = pd.concat([prev, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(leaderboard_path, index=False)
    return run_dir


def write_single_file_master_bundle(
    runs_dir: Path,
    source_mode: str,
    coop_audit_df: pd.DataFrame,
    channel_coverage_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Consolidates EVERY leaderboard row, EVERY stratified slice table, station QC survival audit,
    and channel missingness/gating diagnostics into ONE single JSON file and ONE flat CSV file
    so uploading 1 file gives 100% visibility.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = runs_dir / "leaderboard.csv"
    lb_df = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()

    all_strat_frames: list[pd.DataFrame] = []
    run_details: dict[str, Any] = {}

    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        m_json = sub / "metrics_summary.json"
        m_strat = sub / "metrics_stratified.csv"
        if m_json.exists() and m_strat.exists():
            meta = json.loads(m_json.read_text(encoding="utf-8"))
            sdf = pd.read_csv(m_strat)
            sdf_tagged = sdf.copy()
            sdf_tagged.insert(0, "run_id", sub.name)
            all_strat_frames.append(sdf_tagged)
            run_details[sub.name] = {
                "summary": meta,
                "stratified": sdf.to_dict(orient="records"),
            }

    master_json_path = runs_dir / "master_analysis_bundle.json"
    bundle_obj = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_mode": source_mode,
        "station_qc_audit": coop_audit_df.to_dict(orient="records"),
        "channel_coverage": channel_coverage_df.to_dict(orient="records"),
        "leaderboard": lb_df.to_dict(orient="records"),
        "runs": run_details,
    }
    master_json_path.write_text(json.dumps(bundle_obj, indent=2), encoding="utf-8")

    master_csv_path = runs_dir / "master_all_tables.csv"
    flat_frames: list[pd.DataFrame] = []
    if not lb_df.empty:
        lb_flat = lb_df.copy()
        lb_flat.insert(0, "record_type", "leaderboard")
        flat_frames.append(lb_flat)
    if all_strat_frames:
        st_flat = pd.concat(all_strat_frames, ignore_index=True)
        st_flat.insert(0, "record_type", "stratified")
        flat_frames.append(st_flat)
    if not coop_audit_df.empty:
        aud_flat = coop_audit_df.copy()
        aud_flat.insert(0, "record_type", "station_qc_audit")
        flat_frames.append(aud_flat)

    if flat_frames:
        pd.concat(flat_frames, ignore_index=True).to_csv(master_csv_path, index=False)

    return master_json_path, master_csv_path