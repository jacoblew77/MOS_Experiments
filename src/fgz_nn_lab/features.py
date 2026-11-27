from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MST_OFFSET_HOURS = -7  # Arizona stays on MST (UTC-7) all year
LAPSE_F_PER_M = 0.01168

# Fixed canonical slot ordering so ERA5 pretraining and NWP fine-tuning share exact tensor shapes
PRIMARY_SLOTS = [
    "prim_t2m_min_win",
    "prim_t2m_p10_min_win",
    "prim_t2m_mean_min_win",
    "prim_td2m_min_win",
    "prim_dp_spread_min_win",
    "prim_wind_min_win",
    "prim_wind_med_min_win",
    "prim_pblh_min_win",
    "prim_cloud_min_win",
    "prim_cloud_low_min_win",
    "prim_snow_min_win",
    "prim_t850_min_win",
    "prim_t700_min_win",
    "prim_decouple_idx",
    "prim_t2m_max_cal",
    "prim_t2m_p90_max_cal",
    "prim_t2m_mean_max_cal",
    "prim_t2m_max_coop24",
    "prim_t2m_p90_max_coop24",
    "prim_t2m_min_coop24",
    "prim_td2m_max_cal",
    "prim_dp_spread_max_cal",
    "prim_wind_max_cal",
    "prim_gust_max_cal",
    "prim_pblh_max_cal",
    "prim_sw_rad_max_cal",
    "prim_cloud_max_cal",
    "prim_precip_day",
    "prim_soil_t_max_cal",
    "prim_soil_m_max_cal",
    "prim_t850_max_cal",
    "prim_t700_max_cal",
    "prim_lapse_700_500_max_cal",
]

IFS_SLOTS = [
    "ifs_t2m_min_win",
    "ifs_td2m_min_win",
    "ifs_dp_spread_min_win",
    "ifs_wind_min_win",
    "ifs_cloud_min_win",
    "ifs_snow_min_win",
    "ifs_t850_min_win",
    "ifs_t700_min_win",
    "ifs_t2m_max_cal",
    "ifs_t2m_max_coop24",
    "ifs_td2m_max_cal",
    "ifs_wind_max_cal",
    "ifs_sw_rad_max_cal",
    "ifs_cloud_max_cal",
    "ifs_precip_day",
    "ifs_t850_max_cal",
    "ifs_t700_max_cal",
]

HRRR_SLOTS = [
    "hrrr_t2m_min_win",
    "hrrr_t2m_p10_min_win",
    "hrrr_td2m_min_win",
    "hrrr_dp_spread_min_win",
    "hrrr_wind_min_win",
    "hrrr_pblh_min_win",
    "hrrr_cloud_min_win",
    "hrrr_snow_min_win",
    "hrrr_decouple_idx",
    "hrrr_t2m_max_cal",
    "hrrr_t2m_max_coop24",
    "hrrr_td2m_max_cal",
    "hrrr_wind_max_cal",
    "hrrr_gust_max_cal",
    "hrrr_pblh_max_cal",
    "hrrr_sw_rad_max_cal",
    "hrrr_cloud_max_cal",
]

SPREAD_AND_CAUSAL_SLOTS = [
    "ens_spread_tmin",
    "ens_spread_tmax_cal",
    "ens_spread_tmax_coop24",
    "causal_last_tmin",
    "causal_prev_tmin",
    "causal_mean7_tmin",
    "causal_last_tmax",
    "causal_prev_tmax",
    "causal_mean7_tmax",
    "causal_last_snowd",
]

CANONICAL_DYN_FEATURES = PRIMARY_SLOTS + IFS_SLOTS + HRRR_SLOTS + SPREAD_AND_CAUSAL_SLOTS


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


@dataclass
class TensorBundle:
    issue_dates: np.ndarray            # (N,) datetime64[D]
    station_ids: np.ndarray            # (N,) str
    station_indices: np.ndarray        # (N,) int64 [0 .. num_stations-1]
    station_vocab: list[str]
    tiers: np.ndarray                  # (N,) str
    x_dyn: np.ndarray                  # (N, 5, C_dyn) float32
    mask_dyn: np.ndarray               # (N, 5, C_dyn) float32 (1=valid, 0=missing/masked)
    x_ctx: np.ndarray                  # (N, C_ctx) float32
    raw_anchor: np.ndarray             # (N, 5, 2) float32 default [min_anchor_f, max_cal_anchor_f]
    raw_anchor_candidates: np.ndarray  # (N, 5, 3) float32 [min_f, max_cal_f, max_coop24_f]
    y_true: np.ndarray                 # (N, 5, 2) float32 [tmin_f, tmax_f]
    snow_flag: np.ndarray              # (N, 5) bool
    dyn_feature_names: list[str]
    ctx_feature_names: list[str]


def _aggregate_station_windows(
    hourly_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """
    Aggregates sub-daily steps into BOTH:
      1. Physical nocturnal Min window (00:00..09:00 MST on D) & calendar afternoon Max window (10:00..18:00 MST on D)
      2. True 24-hour COOP morning-reset window ([obs_hour_mst on D-1 .. obs_hour_mst on D]),
         capturing prior-afternoon highs for morning-observer COOP sites (e.g., AZ9359 Winslow, AZ6796 Prescott).
    """
    t0 = time.monotonic()
    obs_hr_map = stations_df.set_index("station_id")["obs_hour_mst"].to_dict()

    df = hourly_df.copy()
    local_time = df["time_utc"] + pd.Timedelta(hours=MST_OFFSET_HOURS)
    df["target_date"] = local_time.dt.floor("D")
    df["hour_mst"] = local_time.dt.hour

    # Shift timestamp by (24 - obs_hour_mst) hours so [obs_hr D-1 .. obs_hr-1 D] lands on date D
    st_obs_hr = df["station_id"].map(obs_hr_map).fillna(7).astype(int)
    coop_shifted = local_time + pd.to_timedelta(24 - st_obs_hr, unit="h")
    df["coop_obs_date"] = coop_shifted.dt.floor("D")

    for col in [
        "dew_point_2m",
        "precipitation",
        "snow_depth",
        "cloud_cover",
        "cloud_cover_low",
        "shortwave_radiation",
        "wind_speed_10m",
        "wind_gusts_10m",
        "boundary_layer_height",
        "soil_temperature_0_to_7cm",
        "soil_moisture_0_to_7cm",
        "temperature_850hPa",
        "temperature_700hPa",
        "temperature_500hPa",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["dp_spread"] = df["temperature_2m"] - df["dew_point_2m"]
    df["lapse_700_500"] = df["temperature_700hPa"] - df["temperature_500hPa"]

    # Nocturnal radiative decoupling proxy: calm wind + clear sky + dry air -> strong basin cooling
    calm_fac = np.clip(1.0 - df["wind_speed_10m"].fillna(8.0) / 12.0, 0.0, 1.0)
    clear_fac = np.clip(1.0 - df["cloud_cover"].fillna(50.0) / 100.0, 0.0, 1.0)
    dry_fac = np.clip(df["dp_spread"].fillna(10.0) / 30.0, 0.0, 1.5)
    df["decouple_idx"] = (calm_fac * clear_fac * dry_fac).astype(np.float32)

    min_win = df[(df["hour_mst"] >= 0) & (df["hour_mst"] <= 9)]
    max_cal = df[(df["hour_mst"] >= 10) & (df["hour_mst"] <= 18)]

    grp_min = min_win.groupby(["station_id", "target_date"], sort=False)
    g_min = grp_min.agg(
        t2m_min_win=("temperature_2m", "min"),
        t2m_mean_min_win=("temperature_2m", "mean"),
        td2m_min_win=("dew_point_2m", "min"),
        dp_spread_min_win=("dp_spread", "min"),
        wind_min_win=("wind_speed_10m", "min"),
        wind_med_min_win=("wind_speed_10m", "median"),
        pblh_min_win=("boundary_layer_height", "min"),
        cloud_min_win=("cloud_cover", "mean"),
        cloud_low_min_win=("cloud_cover_low", "mean"),
        snow_min_win=("snow_depth", "max"),
        t850_min_win=("temperature_850hPa", "mean"),
        t700_min_win=("temperature_700hPa", "mean"),
        decouple_idx=("decouple_idx", "max"),
        grid_elev_m=("grid_elev_m", "first"),
    )
    g_min["t2m_p10_min_win"] = grp_min["temperature_2m"].quantile(0.10)
    g_min = g_min.reset_index()

    grp_max = max_cal.groupby(["station_id", "target_date"], sort=False)
    g_max = grp_max.agg(
        t2m_max_cal=("temperature_2m", "max"),
        t2m_mean_max_cal=("temperature_2m", "mean"),
        td2m_max_cal=("dew_point_2m", "mean"),
        dp_spread_max_cal=("dp_spread", "max"),
        wind_max_cal=("wind_speed_10m", "max"),
        gust_max_cal=("wind_gusts_10m", "max"),
        pblh_max_cal=("boundary_layer_height", "max"),
        sw_rad_max_cal=("shortwave_radiation", "mean"),
        cloud_max_cal=("cloud_cover", "mean"),
        precip_day=("precipitation", "sum"),
        soil_t_max_cal=("soil_temperature_0_to_7cm", "mean"),
        soil_m_max_cal=("soil_moisture_0_to_7cm", "mean"),
        t850_max_cal=("temperature_850hPa", "mean"),
        t700_max_cal=("temperature_700hPa", "mean"),
        lapse_700_500_max_cal=("lapse_700_500", "mean"),
    )
    g_max["t2m_p90_max_cal"] = grp_max["temperature_2m"].quantile(0.90)
    g_max = g_max.reset_index()

    grp_coop24 = df.groupby(["station_id", "coop_obs_date"], sort=False)
    g_coop24 = grp_coop24.agg(
        t2m_max_coop24=("temperature_2m", "max"),
        t2m_min_coop24=("temperature_2m", "min"),
    )
    g_coop24["t2m_p90_max_coop24"] = grp_coop24["temperature_2m"].quantile(0.90)
    g_coop24 = g_coop24.reset_index().rename(columns={"coop_obs_date": "target_date"})

    merged = pd.merge(g_min, g_max, on=["station_id", "target_date"], how="inner")
    merged = pd.merge(merged, g_coop24, on=["station_id", "target_date"], how="left")

    print(
        f"  [{_ts()}] [windows] Aggregated {label:12s} -> {len(merged):,} station-days in {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return merged


def _extract_prev_run_lead_df(prev_df: pd.DataFrame, lead_idx: int) -> pd.DataFrame:
    """Extracts hourly series corresponding to lead_idx in {0,1,2,3,4} (Days 1..5)."""
    suffix = "" if lead_idx == 0 else f"_previous_day{lead_idx}"
    out = pd.DataFrame(
        {
            "station_id": prev_df["station_id"],
            "time_utc": prev_df["time_utc"],
            "grid_elev_m": prev_df["grid_elev_m"],
        }
    )
    base_vars = [
        "temperature_2m",
        "dew_point_2m",
        "precipitation",
        "cloud_cover",
        "shortwave_radiation",
        "wind_speed_10m",
    ]
    for bv in base_vars:
        col = f"{bv}{suffix}"
        out[bv] = prev_df[col] if col in prev_df.columns else np.float32(np.nan)
    return out


def compute_causal_obs_lags(
    coop_df: pd.DataFrame,
    issue_date: pd.Timestamp,
    issue_hour_utc: int,
    morning_min_gate_utc: int = 15,
    afternoon_max_gate_utc: int = 23,
    min_causal_lag_hours: int = 48,
) -> pd.DataFrame:
    issue_ts_utc = pd.Timestamp(issue_date) + pd.Timedelta(hours=issue_hour_utc)
    min_ready_cutoff = issue_ts_utc - pd.Timedelta(hours=max(0, min_causal_lag_hours - morning_min_gate_utc))
    max_ready_cutoff = issue_ts_utc - pd.Timedelta(hours=max(0, min_causal_lag_hours - afternoon_max_gate_utc))

    df = coop_df.copy()
    df["min_gate_ts"] = df["obs_date"] + pd.Timedelta(hours=morning_min_gate_utc)
    df["max_gate_ts"] = df["obs_date"] + pd.Timedelta(hours=afternoon_max_gate_utc)

    valid_min = df[(df["min_gate_ts"] <= issue_ts_utc) & (df["min_gate_ts"] <= min_ready_cutoff)]
    valid_max = df[(df["max_gate_ts"] <= issue_ts_utc) & (df["max_gate_ts"] <= max_ready_cutoff)]

    rows = []
    for sid, _ in df.groupby("station_id"):
        g_min = valid_min[valid_min["station_id"] == sid].sort_values("obs_date")
        g_max = valid_max[valid_max["station_id"] == sid].sort_values("obs_date")

        rows.append(
            {
                "station_id": sid,
                "causal_last_tmin": float(g_min["tmin_f"].iloc[-1]) if len(g_min) >= 1 else np.nan,
                "causal_prev_tmin": float(g_min["tmin_f"].iloc[-2]) if len(g_min) >= 2 else np.nan,
                "causal_mean7_tmin": float(g_min["tmin_f"].tail(7).mean()) if len(g_min) >= 3 else np.nan,
                "causal_last_tmax": float(g_max["tmax_f"].iloc[-1]) if len(g_max) >= 1 else np.nan,
                "causal_prev_tmax": float(g_max["tmax_f"].iloc[-2]) if len(g_max) >= 2 else np.nan,
                "causal_mean7_tmax": float(g_max["tmax_f"].tail(7).mean()) if len(g_max) >= 3 else np.nan,
                "causal_last_snowd": float(g_min["snowd"].iloc[-1]) if len(g_min) >= 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _merge_canonical_day_table(
    prim_df: pd.DataFrame,
    era5_df: pd.DataFrame | None,
    ifs_df: pd.DataFrame | None,
    hrrr_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Builds a unified daily DataFrame matching CANONICAL_DYN_FEATURES (excluding causal lags)."""
    base = prim_df.copy()
    if era5_df is not None and era5_df is not prim_df:
        era5_sub = era5_df[
            ["station_id", "target_date", "soil_t_max_cal", "soil_m_max_cal", "snow_min_win", "pblh_min_win", "pblh_max_cal"]
        ].rename(
            columns={
                "soil_t_max_cal": "_era5_soil_t",
                "soil_m_max_cal": "_era5_soil_m",
                "snow_min_win": "_era5_snow",
                "pblh_min_win": "_era5_pbl_min",
                "pblh_max_cal": "_era5_pbl_max",
            }
        )
        base = pd.merge(base, era5_sub, on=["station_id", "target_date"], how="left")
        base["soil_t_max_cal"] = base["soil_t_max_cal"].fillna(base["_era5_soil_t"])
        base["soil_m_max_cal"] = base["soil_m_max_cal"].fillna(base["_era5_soil_m"])
        base["snow_min_win"] = base["snow_min_win"].fillna(base["_era5_snow"])
        base["pblh_min_win"] = base["pblh_min_win"].fillna(base["_era5_pbl_min"])
        base["pblh_max_cal"] = base["pblh_max_cal"].fillna(base["_era5_pbl_max"])

    out = pd.DataFrame(
        {
            "station_id": base["station_id"],
            "target_date": base["target_date"],
            "prim_grid_elev_m": base["grid_elev_m"],
        }
    )
    for slot in PRIMARY_SLOTS:
        raw_col = slot.replace("prim_", "", 1)
        out[slot] = base[raw_col] if raw_col in base.columns else np.nan

    if ifs_df is not None:
        ifs_ren = {
            slot.replace("ifs_", "", 1): slot for slot in IFS_SLOTS
        }
        ifs_keep = ["station_id", "target_date"] + [k for k in ifs_ren if k in ifs_df.columns]
        ifs_sub = ifs_df[ifs_keep].rename(columns=ifs_ren)
        out = pd.merge(out, ifs_sub, on=["station_id", "target_date"], how="left")
    for slot in IFS_SLOTS:
        if slot not in out.columns:
            out[slot] = np.nan

    if hrrr_df is not None:
        hrrr_ren = {
            slot.replace("hrrr_", "", 1): slot for slot in HRRR_SLOTS
        }
        hrrr_keep = ["station_id", "target_date", "grid_elev_m"] + [k for k in hrrr_ren if k in hrrr_df.columns]
        hrrr_sub = hrrr_df[hrrr_keep].rename(columns={**hrrr_ren, "grid_elev_m": "hrrr_grid_elev_m"})
        out = pd.merge(out, hrrr_sub, on=["station_id", "target_date"], how="left")
    else:
        out["hrrr_grid_elev_m"] = np.nan
    for slot in HRRR_SLOTS:
        if slot not in out.columns:
            out[slot] = np.nan

    if ifs_df is not None:
        out["ens_spread_tmin"] = (out["prim_t2m_min_win"] - out["ifs_t2m_min_win"]).abs()
        out["ens_spread_tmax_cal"] = (out["prim_t2m_max_cal"] - out["ifs_t2m_max_cal"]).abs()
        out["ens_spread_tmax_coop24"] = (out["prim_t2m_max_coop24"] - out["ifs_t2m_max_coop24"]).abs()
    else:
        out["ens_spread_tmin"] = np.nan
        out["ens_spread_tmax_cal"] = np.nan
        out["ens_spread_tmax_coop24"] = np.nan

    return out


def build_aligned_tensors(
    stations_df: pd.DataFrame,
    coop_df: pd.DataFrame,
    hourly_by_source: dict[str, pd.DataFrame],
    issue_hour_utc: int = 12,
    morning_min_gate_utc: int = 15,
    afternoon_max_gate_utc: int = 23,
    min_causal_lag_hours: int = 48,
    station_vocab: list[str] | None = None,
) -> TensorBundle:
    t_start = time.monotonic()
    has_prev_runs = "prev_gfs" in hourly_by_source

    era5_daily = (
        _aggregate_station_windows(hourly_by_source["era5"], stations_df, "ERA5")
        if "era5" in hourly_by_source
        else None
    )
    gfs_hist_daily = (
        _aggregate_station_windows(hourly_by_source["gfs"], stations_df, "GFS_HIST")
        if "gfs" in hourly_by_source
        else None
    )
    ifs_hist_daily = (
        _aggregate_station_windows(hourly_by_source["ifs"], stations_df, "IFS_HIST")
        if "ifs" in hourly_by_source
        else None
    )
    hrrr_hist_daily = (
        _aggregate_station_windows(hourly_by_source["hrrr"], stations_df, "HRRR_HIST")
        if "hrrr" in hourly_by_source
        else None
    )

    # Build lead-specific canonical tables per lead_idx d in 0..4
    lead_tables: list[pd.DataFrame] = []
    if has_prev_runs:
        for d_idx in range(5):
            pgfs_d = _aggregate_station_windows(
                _extract_prev_run_lead_df(hourly_by_source["prev_gfs"], d_idx),
                stations_df,
                f"PREV_GFS_D{d_idx + 1}",
            )
            # Backfill upper-air & PBL fields not exposed on previous-runs-api from GFS_HIST if Day 1, else NaN
            if gfs_hist_daily is not None and d_idx == 0:
                extra_u = gfs_hist_daily[
                    ["station_id", "target_date", "pblh_min_win", "pblh_max_cal", "t850_min_win", "t700_min_win", "t850_max_cal", "t700_max_cal", "lapse_700_500_max_cal", "snow_min_win"]
                ]
                pgfs_d = pgfs_d.drop(columns=[c for c in extra_u.columns if c in pgfs_d.columns and c not in ("station_id", "target_date")])
                pgfs_d = pd.merge(pgfs_d, extra_u, on=["station_id", "target_date"], how="left")

            pifs_d = None
            if "prev_ifs" in hourly_by_source:
                pifs_d = _aggregate_station_windows(
                    _extract_prev_run_lead_df(hourly_by_source["prev_ifs"], d_idx),
                    stations_df,
                    f"PREV_IFS_D{d_idx + 1}",
                )
            phrrr_d = None
            if d_idx < 2 and "prev_hrrr" in hourly_by_source:
                phrrr_d = _aggregate_station_windows(
                    _extract_prev_run_lead_df(hourly_by_source["prev_hrrr"], d_idx),
                    stations_df,
                    f"PREV_HRRR_D{d_idx + 1}",
                )

            # Never leak future ERA5 soil/snow on Days 2-5 in true_lead_nwp mode
            lead_tables.append(
                _merge_canonical_day_table(
                    prim_df=pgfs_d,
                    era5_df=era5_daily if d_idx == 0 else None,
                    ifs_df=pifs_d,
                    hrrr_df=phrrr_d,
                )
            )
    else:
        prim_base = gfs_hist_daily if gfs_hist_daily is not None else era5_daily
        assert prim_base is not None
        shared_tbl = _merge_canonical_day_table(
            prim_df=prim_base,
            era5_df=era5_daily,
            ifs_df=ifs_hist_daily,
            hrrr_df=hrrr_hist_daily,
        )
        lead_tables = [shared_tbl for _ in range(5)]

    non_causal_cols = PRIMARY_SLOTS + IFS_SLOTS + HRRR_SLOTS + [
        "ens_spread_tmin",
        "ens_spread_tmax_cal",
        "ens_spread_tmax_coop24",
    ]
    causal_cols = [
        "causal_last_tmin",
        "causal_prev_tmin",
        "causal_mean7_tmin",
        "causal_last_tmax",
        "causal_prev_tmax",
        "causal_mean7_tmax",
        "causal_last_snowd",
    ]
    all_dyn_names = non_causal_cols + causal_cols
    assert all_dyn_names == CANONICAL_DYN_FEATURES

    st_meta = stations_df.set_index("station_id")
    vocab = station_vocab if station_vocab is not None else sorted(stations_df["station_id"].tolist())
    sid_to_idx = {sid: i for i, sid in enumerate(vocab)}

    tier_codes = {
        "basin_plateau": [1.0, 0.0, 0.0],
        "rim_slope": [0.0, 1.0, 0.0],
        "canyon": [0.0, 0.0, 1.0],
    }

    coop_by_sid = {
        sid: g.sort_values("obs_date").reset_index(drop=True)
        for sid, g in coop_df.groupby("station_id")
    }
    lead_by_sid: list[dict[str, pd.DataFrame]] = [
        {sid: g.sort_values("target_date").reset_index(drop=True) for sid, g in tbl.groupby("station_id")}
        for tbl in lead_tables
    ]

    min_lag_days_min = int(np.ceil(max(1, (min_causal_lag_hours + morning_min_gate_utc - issue_hour_utc) / 24.0)))
    min_lag_days_max = int(np.ceil(max(1, (min_causal_lag_hours + afternoon_max_gate_utc - issue_hour_utc) / 24.0)))

    hrrr_col_indices = [i for i, name in enumerate(all_dyn_names) if name.startswith("hrrr_")]
    idx_prim_min = non_causal_cols.index("prim_t2m_min_win")
    idx_prim_max_cal = non_causal_cols.index("prim_t2m_max_cal")
    idx_prim_max_c24 = non_causal_cols.index("prim_t2m_max_coop24")
    idx_prim_snow = non_causal_cols.index("prim_snow_min_win")
    idx_hrrr_min = non_causal_cols.index("hrrr_t2m_min_win")
    idx_hrrr_max_cal = non_causal_cols.index("hrrr_t2m_max_cal")
    idx_hrrr_max_c24 = non_causal_cols.index("hrrr_t2m_max_coop24")

    issue_dates_out = []
    sids_out = []
    sidx_out = []
    tiers_out = []
    x_dyn_list = []
    mask_dyn_list = []
    x_ctx_list = []
    anchor_default_list = []
    anchor_cand_list = []
    y_list = []
    snow_list = []

    active_sids = [sid for sid in st_meta.index if sid in coop_by_sid and sid in lead_by_sid[0]]
    total_st = len(active_sids)

    for st_i, sid in enumerate(active_sids, start=1):
        st_t0 = time.monotonic()
        st_row = st_meta.loc[sid]
        elev_m = float(st_row["elev_m"])
        lat = float(st_row["lat"])
        lon = float(st_row["lon"])
        obs_hr = float(st_row["obs_hour_mst"])
        tier_str = str(st_row["tier"])
        tier_oh = tier_codes.get(tier_str, [1.0, 0.0, 0.0])
        st_int_id = int(sid_to_idx.get(sid, 0))

        st_coop = coop_by_sid[sid]
        coop_dates = st_coop["obs_date"].values.astype("datetime64[D]")
        coop_tmin = st_coop["tmin_f"].to_numpy(dtype=np.float32)
        coop_tmax = st_coop["tmax_f"].to_numpy(dtype=np.float32)
        coop_snowd = st_coop["snowd"].to_numpy(dtype=np.float32)
        coop_map = {d: idx for idx, d in enumerate(coop_dates)}

        d_maps = []
        d_mats = []
        prim_grid_elev = elev_m
        hrrr_grid_elev = elev_m

        for d_idx in range(5):
            sdf = lead_by_sid[d_idx].get(sid)
            if sdf is None or sdf.empty:
                d_maps.append({})
                d_mats.append(np.empty((0, len(non_causal_cols)), dtype=np.float32))
                continue
            if d_idx == 0:
                if sdf["prim_grid_elev_m"].notna().any():
                    prim_grid_elev = float(sdf["prim_grid_elev_m"].dropna().iloc[0])
                if sdf["hrrr_grid_elev_m"].notna().any():
                    hrrr_grid_elev = float(sdf["hrrr_grid_elev_m"].dropna().iloc[0])
                else:
                    hrrr_grid_elev = prim_grid_elev
            dd = sdf["target_date"].values.astype("datetime64[D]")
            d_maps.append({dt_val: i for i, dt_val in enumerate(dd)})
            d_mats.append(sdf[non_causal_cols].to_numpy(dtype=np.float32))

        elev_delta_m = elev_m - prim_grid_elev
        hrrr_delta_m = elev_m - hrrr_grid_elev

        st_samples = 0
        for issue_d64 in coop_dates:
            horizon_d64 = [issue_d64 + np.timedelta64(d, "D") for d in range(5)]
            if not all((hd in coop_map and hd in d_maps[d_i]) for d_i, hd in enumerate(horizon_d64)):
                continue

            cutoff_min_d64 = issue_d64 - np.timedelta64(min_lag_days_min, "D")
            cutoff_max_d64 = issue_d64 - np.timedelta64(min_lag_days_max, "D")
            idx_min = int(np.searchsorted(coop_dates, cutoff_min_d64, side="right")) - 1
            idx_max = int(np.searchsorted(coop_dates, cutoff_max_d64, side="right")) - 1

            c_vals = np.full(len(causal_cols), np.nan, dtype=np.float32)
            if idx_min >= 2:
                c_vals[0] = coop_tmin[idx_min]
                c_vals[1] = coop_tmin[idx_min - 1]
                c_vals[2] = float(coop_tmin[max(0, idx_min - 6) : idx_min + 1].mean())
                c_vals[6] = coop_snowd[idx_min]
            if idx_max >= 2:
                c_vals[3] = coop_tmax[idx_max]
                c_vals[4] = coop_tmax[idx_max - 1]
                c_vals[5] = float(coop_tmax[max(0, idx_max - 6) : idx_max + 1].mean())

            y_sample = np.zeros((5, 2), dtype=np.float32)
            dyn_sample = np.full((5, len(all_dyn_names)), np.nan, dtype=np.float32)
            anc_cand = np.zeros((5, 3), dtype=np.float32)
            snow_sample = np.zeros(5, dtype=bool)
            valid_sample = True

            for d_idx, hd in enumerate(horizon_d64):
                ci = coop_map[hd]
                di = d_maps[d_idx][hd]
                row_vals = d_mats[d_idx][di]

                y_sample[d_idx, 0] = coop_tmin[ci]
                y_sample[d_idx, 1] = coop_tmax[ci]

                dyn_sample[d_idx, : len(non_causal_cols)] = row_vals
                dyn_sample[d_idx, len(non_causal_cols) :] = c_vals

                if d_idx >= 2 and hrrr_col_indices:
                    dyn_sample[d_idx, hrrr_col_indices] = np.nan

                # Compute [min_anchor, max_cal_anchor, max_coop24_anchor]
                use_hrrr = (
                    d_idx < 2
                    and not np.isnan(row_vals[idx_hrrr_min])
                    and not np.isnan(row_vals[idx_hrrr_max_cal])
                )
                if use_hrrr:
                    r_min = float(row_vals[idx_hrrr_min]) - LAPSE_F_PER_M * hrrr_delta_m
                    r_max_cal = float(row_vals[idx_hrrr_max_cal]) - LAPSE_F_PER_M * hrrr_delta_m
                    r_c24_val = row_vals[idx_hrrr_max_c24]
                    r_max_c24 = (
                        float(r_c24_val) - LAPSE_F_PER_M * hrrr_delta_m
                        if not np.isnan(r_c24_val)
                        else r_max_cal
                    )
                else:
                    r_min = float(row_vals[idx_prim_min]) - LAPSE_F_PER_M * elev_delta_m
                    r_max_cal = float(row_vals[idx_prim_max_cal]) - LAPSE_F_PER_M * elev_delta_m
                    r_c24_val = row_vals[idx_prim_max_c24]
                    r_max_c24 = (
                        float(r_c24_val) - LAPSE_F_PER_M * elev_delta_m
                        if not np.isnan(r_c24_val)
                        else r_max_cal
                    )

                if np.isnan(r_min) or np.isnan(r_max_cal):
                    valid_sample = False
                    break

                anc_cand[d_idx, 0] = r_min
                anc_cand[d_idx, 1] = r_max_cal
                anc_cand[d_idx, 2] = r_max_c24

                mod_snow = float(row_vals[idx_prim_snow]) if not np.isnan(row_vals[idx_prim_snow]) else 0.0
                snow_sample[d_idx] = (coop_snowd[ci] > 0.1) or (mod_snow > 0.01)

            if not valid_sample:
                continue

            mask_sample = (~np.isnan(dyn_sample)).astype(np.float32)
            dyn_clean = np.nan_to_num(dyn_sample, nan=0.0)

            doy = pd.Timestamp(issue_d64).dayofyear
            sin_doy = float(np.sin(2.0 * np.pi * doy / 365.25))
            cos_doy = float(np.cos(2.0 * np.pi * doy / 365.25))

            ctx_vec = np.array(
                [
                    elev_m / 1000.0,
                    elev_delta_m / 1000.0,
                    (lat - 35.0),
                    (lon + 111.5),
                    obs_hr / 24.0,
                    sin_doy,
                    cos_doy,
                    *tier_oh,
                ],
                dtype=np.float32,
            )

            issue_dates_out.append(issue_d64)
            sids_out.append(sid)
            sidx_out.append(st_int_id)
            tiers_out.append(tier_str)
            x_dyn_list.append(dyn_clean)
            mask_dyn_list.append(mask_sample)
            x_ctx_list.append(ctx_vec)
            anchor_default_list.append(anc_cand[:, :2].copy())
            anchor_cand_list.append(anc_cand)
            y_list.append(y_sample)
            snow_list.append(snow_sample)
            st_samples += 1

        print(
            f"  [{_ts()}] [tensors {st_i:02d}/{total_st:02d}] {sid} ({st_row['name']}): "
            f"{st_samples:,} 5-day sequences aligned in {time.monotonic() - st_t0:.2f}s",
            flush=True,
        )

    ctx_names = [
        "elev_km",
        "elev_delta_km",
        "lat_centered",
        "lon_centered",
        "obs_hour_norm",
        "sin_doy",
        "cos_doy",
        "tier_basin",
        "tier_rim",
        "tier_canyon",
    ]

    order = np.argsort(np.array(issue_dates_out))
    print(
        f"  [{_ts()}] [tensors] Total aligned dataset ready ({len(order):,} samples, {len(all_dyn_names)} channels) "
        f"in {time.monotonic() - t_start:.1f}s",
        flush=True,
    )
    return TensorBundle(
        issue_dates=np.array(issue_dates_out)[order],
        station_ids=np.array(sids_out)[order],
        station_indices=np.array(sidx_out, dtype=np.int64)[order],
        station_vocab=vocab,
        tiers=np.array(tiers_out)[order],
        x_dyn=np.stack(x_dyn_list, axis=0)[order],
        mask_dyn=np.stack(mask_dyn_list, axis=0)[order],
        x_ctx=np.stack(x_ctx_list, axis=0)[order],
        raw_anchor=np.stack(anchor_default_list, axis=0)[order],
        raw_anchor_candidates=np.stack(anchor_cand_list, axis=0)[order],
        y_true=np.stack(y_list, axis=0)[order],
        snow_flag=np.stack(snow_list, axis=0)[order],
        dyn_feature_names=all_dyn_names,
        ctx_feature_names=ctx_names,
    )