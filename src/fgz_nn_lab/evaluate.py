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
    """Computes headline 5-day MAE, per-lead-day MAE, and FGZ regime/station slices."""
    abs_err = np.abs(y_pred - y_true)  # (N, 5, 2)
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

    # Freezing night slice (Observed Min T <= 32 F)
    freeze_mask = y_true[:, :, 0] <= 32.0
    if freeze_mask.any():
        slice_rows.append(
            {
                "slice_type": "regime",
                "slice_name": "freezing_nights_le_32F",
                "samples": int(freeze_mask.sum()),
                "mae_min": float(min_err[freeze_mask].mean()),
                "mae_max": float(max_err[freeze_mask].mean()),
            }
        )

    # Snow cover slice
    if snow_flag.any():
        slice_rows.append(
            {
                "slice_type": "regime",
                "slice_name": "snow_present",
                "samples": int(snow_flag.sum()),
                "mae_min": float(min_err[snow_flag].mean()),
                "mae_max": float(max_err[snow_flag].mean()),
            }
        )

    # Microclimate tier slices
    for tier in np.unique(tiers):
        m = tiers == tier
        slice_rows.append(
            {
                "slice_type": "tier",
                "slice_name": str(tier),
                "samples": int(m.sum() * 5),
                "mae_min": float(min_err[m].mean()),
                "mae_max": float(max_err[m].mean()),
            }
        )

    # Per-station slices
    for sid in np.unique(station_ids):
        m = station_ids == sid
        slice_rows.append(
            {
                "slice_type": "station",
                "slice_name": str(sid),
                "samples": int(m.sum() * 5),
                "mae_min": float(min_err[m].mean()),
                "mae_max": float(max_err[m].mean()),
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
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics_summary.json").write_text(
        json.dumps({"val": val_summary, "test": test_summary}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config_snapshot, indent=2, default=str),
        encoding="utf-8",
    )
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
        combined = pd.concat([prev, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(leaderboard_path, index=False)
    return run_dir