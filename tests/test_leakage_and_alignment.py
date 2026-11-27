from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fgz_nn_lab.dashboard import build_github_pages_site
from fgz_nn_lab.data_ingest import filter_coop_qc, load_station_catalog
from fgz_nn_lab.features import build_aligned_tensors, compute_causal_obs_lags
from fgz_nn_lab.models import (
    apply_network_aware_blend,
    fit_conformal_residual_quantiles,
    fit_network_aware_blend_weights,
)
from fgz_nn_lab.splits import FoldScaler, make_purged_expanding_splits


def test_station_catalog_phantom_ranch_and_26_seeds() -> None:
    df = load_station_catalog(Path("configs/stations_fgz_coop.json"), raw_dir=None)
    pr = df[df["station_id"] == "PHRA3"].iloc[0]
    assert pr["elev_ft"] == 2530.0
    assert "KPAN" in df["uid"].values
    assert "KSEZ" in df["uid"].values
    assert "AZ0000" not in df["uid"].values


def test_strict_real_obs_qc_rejects_estimated_and_blanks() -> None:
    raw = pd.DataFrame(
        {
            "station": ["FLG", "FGZ", "AZ0000", "AZC001", "PHRA3"],
            "day": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-01", "2025-01-01"],
            "max_temp_f": [45.0, 46.0, 50.0, 52.0, ""],
            "min_temp_f": [18.0, 19.0, 20.0, 21.0, 38.0],
            "temp_estimated": ["False", "True", "False", "False", "False"],
        }
    )
    qc = filter_coop_qc(raw)
    assert list(qc["station"]) == ["FLG"]


def test_network_aware_stack_and_conformal_quantiles(tmp_path: Path) -> None:
    dates = pd.date_range("2025-03-01", "2025-03-10", freq="D")
    coop = pd.DataFrame(
        {
            "station_id": ["KFLG"] * len(dates),
            "obs_date": dates,
            "tmin_f": np.arange(10, 10 + len(dates), dtype=float),
            "tmax_f": np.arange(40, 40 + len(dates), dtype=float),
            "snowd": np.zeros(len(dates)),
        }
    )
    lags = compute_causal_obs_lags(coop, pd.Timestamp("2025-03-08"), 12, 15, 23, 48)
    assert lags["causal_last_tmin"].iloc[0] <= 15.0

    y = np.zeros((100, 5, 2), dtype=np.float32)
    nets = np.array(["AZ_ASOS"] * 50 + ["AZ_COOP"] * 50)
    p_nn = np.ones((100, 5, 2), dtype=np.float32) * 2.5
    p_lg = np.ones((100, 5, 2), dtype=np.float32) * 0.1
    p_nn[50:] = 0.1  # NN wins on COOP
    p_lg[50:] = 2.5  # LGBM wins on ASOS

    w_map = fit_network_aware_blend_weights(p_nn, p_lg, y, nets)
    assert np.allclose(w_map["AZ_ASOS"], 0.0)
    assert np.allclose(w_map["AZ_COOP"], 1.0)
    blended = apply_network_aware_blend(p_nn, p_lg, nets, w_map)
    assert np.allclose(blended, 0.1)

    q_map = fit_conformal_residual_quantiles(blended, y, nets)
    assert q_map["AZ_COOP"]["q10"].shape == (5, 2)

    st_df = pd.DataFrame([{"station_id": "KFLG", "name": "Flagstaff", "lat": 35.14, "lon": -111.67, "elev_ft": 7011.0, "tier": "basin_plateau", "network": "AZ_ASOS"}])
    live = [{"station_id": "KFLG", "name": "Flagstaff", "network": "AZ_ASOS", "forecasts": [{"day": 1, "date": "2026-11-27", "tmin_f": 25.0, "tmin_p10_f": 21.5, "tmin_p90_f": 28.5, "tmax_f": 55.0, "tmax_p10_f": 52.5, "tmax_p90_f": 57.5, "nn_tmin_f": 24.8, "lgbm_tmin_f": 25.2}]}]
    idx_html = build_github_pages_site(tmp_path, st_df, live, [], {"KFLG": {"mae_min": 2.4, "mae_max": 1.5}})
    assert idx_html.exists()
    assert (tmp_path / "forecasts_latest.csv").exists()


def test_sparse_channel_guard_and_purged_splits() -> None:
    st = pd.DataFrame(
        [
            {
                "station_id": "KFLG",
                "network": "AZ_ASOS",
                "mos_id": "KFLG",
                "name": "Flagstaff",
                "lat": 35.14,
                "lon": -111.67,
                "elev_ft": 7011.0,
                "elev_m": 2137.0,
                "tier": "basin_plateau",
                "obs_hour_mst": 0,
            }
        ]
    )
    days = pd.date_range("2025-01-01", "2025-03-10", freq="D")
    coop = pd.DataFrame(
        {
            "station_id": "KFLG",
            "obs_date": days,
            "tmin_f": 20.0,
            "tmax_f": 48.0,
            "precip": 0.0,
            "snow": 0.0,
            "snowd": 0.0,
        }
    )
    hours = pd.date_range("2024-12-30", "2025-03-18", freq="h")
    hourly = pd.DataFrame(
        {
            "station_id": "KFLG",
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
    bundle = build_aligned_tensors(st, coop, {"era5": hourly, "hrrr": hourly.copy()})
    t2m_hrrr_col = bundle.dyn_feature_names.index("hrrr_t2m_min_win")
    assert (bundle.mask_dyn[:, :2, t2m_hrrr_col] == 1.0).all()
    assert (bundle.mask_dyn[:, 2:, t2m_hrrr_col] == 0.0).all()

    folds = make_purged_expanding_splits(bundle.issue_dates, test_days=12, val_folds=1, val_fold_days=12, purge_days=6)
    f0 = folds[0]
    assert (bundle.issue_dates[f0.val_idx].min() - bundle.issue_dates[f0.train_idx].max()) >= np.timedelta64(6, "D")
    scaler = FoldScaler(min_valid_samples=10).fit(
        bundle.x_dyn[f0.train_idx], bundle.mask_dyn[f0.train_idx], bundle.x_ctx[f0.train_idx],
        bundle.raw_anchor[f0.train_idx], bundle.y_true[f0.train_idx], bundle.station_ids[f0.train_idx],
        bundle.raw_anchor_candidates[f0.train_idx],
    )
    z_te, _, _ = scaler.transform_inputs(bundle.x_dyn[f0.test_idx], bundle.mask_dyn[f0.test_idx], bundle.x_ctx[f0.test_idx])
    assert np.abs(z_te).max() <= 8.0