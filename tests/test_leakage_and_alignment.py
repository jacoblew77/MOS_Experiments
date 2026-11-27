from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fgz_nn_lab.data_ingest import filter_coop_qc, load_station_catalog
from fgz_nn_lab.features import build_aligned_tensors, compute_causal_obs_lags
from fgz_nn_lab.splits import FoldScaler, make_purged_expanding_splits


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
    lags = compute_causal_obs_lags(
        coop,
        issue_date=pd.Timestamp("2025-03-08"),
        issue_hour_utc=12,
        morning_min_gate_utc=15,
        afternoon_max_gate_utc=23,
        min_causal_lag_hours=48,
    )
    assert lags["causal_last_tmin"].iloc[0] <= 15.0
    assert lags["causal_last_tmax"].iloc[0] <= 45.0


def test_sparse_channel_guard_prevents_zscore_explosion() -> None:
    n_tr, n_te, c_dim = 200, 20, 4
    x_tr = np.ones((n_tr, 5, c_dim), dtype=np.float32) * 40.0
    x_tr[:, :, 0] += np.linspace(-5.0, 5.0, n_tr, dtype=np.float32)[:, None]
    m_tr = np.ones((n_tr, 5, c_dim), dtype=np.float32)
    # Channel 3 is completely absent in training (e.g. HRRR prior to Sep 2025)
    x_tr[:, :, 3] = 0.0
    m_tr[:, :, 3] = 0.0

    x_te = np.ones((n_te, 5, c_dim), dtype=np.float32) * 40.0
    x_te[:, :, 3] = 1500.0  # Turns on in test with huge raw PBL magnitude
    m_te = np.ones((n_te, 5, c_dim), dtype=np.float32)

    ctx_tr = np.random.randn(n_tr, 5).astype(np.float32)
    ctx_te = np.random.randn(n_te, 5).astype(np.float32)
    anc_tr = np.ones((n_tr, 5, 2), dtype=np.float32) * 40.0
    y_tr = np.ones((n_tr, 5, 2), dtype=np.float32) * 41.0
    sids_tr = np.array(["AZ3010"] * n_tr)

    scaler = FoldScaler(min_valid_samples=150, clip_zscore=8.0).fit(
        x_tr, m_tr, ctx_tr, anc_tr, y_tr, sids_tr
    )
    z_te, eff_m_te, _ = scaler.transform_inputs(x_te, m_te, ctx_te)
    assert scaler.active_channel_mask[3] == 0.0
    assert (eff_m_te[:, :, 3] == 0.0).all()
    assert (z_te[:, :, 3] == 0.0).all()
    assert np.abs(z_te).max() <= 8.0


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
    days = pd.date_range("2025-01-01", "2025-03-01", freq="D")
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
    hours = pd.date_range("2024-12-30", "2025-03-08", freq="h")
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
    t2m_hrrr_col = bundle.dyn_feature_names.index("hrrr_t2m_min_win")
    assert (bundle.mask_dyn[:, :2, t2m_hrrr_col] == 1.0).all()
    assert (bundle.mask_dyn[:, 2:, t2m_hrrr_col] == 0.0).all()

    folds = make_purged_expanding_splits(bundle.issue_dates, test_days=12, val_folds=1, val_fold_days=12, purge_days=6)
    f0 = folds[0]
    assert (bundle.issue_dates[f0.val_idx].min() - bundle.issue_dates[f0.train_idx].max()) >= np.timedelta64(6, "D")
    assert (bundle.issue_dates[f0.test_idx].min() - bundle.issue_dates[f0.val_idx].max()) >= np.timedelta64(6, "D")