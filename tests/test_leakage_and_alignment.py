from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fgz_nn_lab.data_ingest import filter_coop_qc, load_station_catalog
from fgz_nn_lab.features import build_aligned_tensors, compute_causal_obs_lags
from fgz_nn_lab.splits import make_purged_expanding_splits


def test_station_catalog_phantom_ranch_override() -> None:
    df = load_station_catalog(Path("configs/stations_fgz_coop.json"))
    pr = df[df["station_id"] == "AZ6471"].iloc[0]
    assert pr["elev_ft"] == 2530.0
    assert "AZ0000" not in df["station_id"].values
    assert not df["station_id"].str.startswith("AZC").any()


def test_coop_qc_filters_estimated_and_synthetic() -> None:
    raw = pd.DataFrame(
        {
            "station": ["AZ3010", "AZ3010", "AZ0000", "AZC001", "AZ6471"],
            "day": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-01", "2025-01-01"],
            "high": [45.0, 46.0, 50.0, 52.0, 62.0],
            "low": [18.0, 19.0, 20.0, 21.0, 38.0],
            "temp_estimated": ["False", "True", "False", "False", False],
        }
    )
    qc = filter_coop_qc(raw)
    assert list(qc["station"]) == ["AZ3010", "AZ6471"]


def test_gating_and_causal_lag_never_leak_future() -> None:
    dates = pd.date_range("2025-03-01", "2025-03-10", freq="D")
    coop = pd.DataFrame(
        {
            "station_id": ["AZ3010"] * len(dates),
            "obs_date": dates,
            "tmin_f": np.arange(10, 10 + len(dates), dtype=float),
            "tmax_f": np.arange(40, 40 + len(dates), dtype=float),
            "snowd": np.zeros(len(dates)),
        }
    )
    # Issue at 2025-03-08 12:00Z (before 15:00Z Morning Min gate and with >=48h lag rule)
    lags = compute_causal_obs_lags(
        coop,
        issue_date=pd.Timestamp("2025-03-08"),
        issue_hour_utc=12,
        morning_min_gate_utc=15,
        afternoon_max_gate_utc=23,
        min_causal_lag_hours=48,
    )
    # Latest allowed Morning Min must be <= 2025-03-06 (value 15.0, never March 7 or March 8)
    assert lags["causal_last_tmin"].iloc[0] <= 15.0
    assert lags["causal_last_tmax"].iloc[0] <= 45.0


def test_hrrr_masked_on_days_3_to_5_and_split_purged() -> None:
    st = pd.DataFrame(
        [
            {
                "station_id": "AZ3010",
                "name": "Flagstaff",
                "lat": 35.14,
                "lon": -111.67,
                "elev_ft": 7003.0,
                "elev_m": 2134.5,
                "tier": "basin_plateau",
                "obs_hour_mst": 7,
            }
        ]
    )
    days = pd.date_range("2025-01-01", "2025-02-15", freq="D")
    coop = pd.DataFrame(
        {
            "station_id": "AZ3010",
            "obs_date": days,
            "tmin_f": 20.0,
            "tmax_f": 48.0,
            "precip": 0.0,
            "snow": 0.0,
            "snowd": 0.0,
        }
    )
    hours = pd.date_range("2025-01-01", "2025-02-20", freq="h")
    hourly = pd.DataFrame(
        {
            "station_id": "AZ3010",
            "time_utc": hours,
            "temperature_2m": 34.0,
            "dew_point_2m": 18.0,
            "precipitation": 0.0,
            "snow_depth": 0.0,
            "cloud_cover": 20.0,
            "shortwave_radiation": 250.0,
            "wind_speed_10m": 6.0,
            "grid_elev_m": 2100.0,
        }
    )
    bundle = build_aligned_tensors(
        stations_df=st,
        coop_df=coop,
        hourly_by_source={"era5": hourly, "hrrr": hourly.copy()},
    )
    hrrr_idx = [i for i, n in enumerate(bundle.dyn_feature_names) if n.startswith("hrrr_")]
    assert len(hrrr_idx) > 0
    # Days 1-2 (indices 0, 1) have valid HRRR temperature mask == 1; Days 3-5 (2, 3, 4) must be 0
    t2m_hrrr_col = bundle.dyn_feature_names.index("hrrr_t2m_min_win")
    assert (bundle.mask_dyn[:, :2, t2m_hrrr_col] == 1.0).all()
    assert (bundle.mask_dyn[:, 2:, t2m_hrrr_col] == 0.0).all()

    folds = make_purged_expanding_splits(bundle.issue_dates, test_days=10, val_folds=1, val_fold_days=10, purge_days=6)
    f0 = folds[0]
    max_tr_day = bundle.issue_dates[f0.train_idx].max()
    min_va_day = bundle.issue_dates[f0.val_idx].min()
    max_va_day = bundle.issue_dates[f0.val_idx].max()
    min_te_day = bundle.issue_dates[f0.test_idx].min()
    assert (min_va_day - max_tr_day) >= np.timedelta64(6, "D")
    assert (min_te_day - max_va_day) >= np.timedelta64(6, "D")