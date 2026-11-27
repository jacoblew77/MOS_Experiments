from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MST_OFFSET_HOURS = -7  # Arizona stays on MST (UTC-7) all year


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


@dataclass
class TensorBundle:
    issue_dates: np.ndarray        # (N,) datetime64[D]
    station_ids: np.ndarray        # (N,) str
    tiers: np.ndarray              # (N,) str
    x_dyn: np.ndarray              # (N, 5, C_dyn) float32
    mask_dyn: np.ndarray           # (N, 5, C_dyn) float32 (1=valid, 0=missing/masked)
    x_ctx: np.ndarray              # (N, C_ctx) float32
    raw_anchor: np.ndarray         # (N, 5, 2) float32 [min_anchor_f, max_anchor_f]
    y_true: np.ndarray             # (N, 5, 2) float32 [tmin_f, tmax_f]
    snow_flag: np.ndarray          # (N, 5) bool
    dyn_feature_names: list[str]
    ctx_feature_names: list[str]


def _aggregate_daily_windows(hourly_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Vectorized aggregation of sub-daily hourly NWP/reanalysis steps into MST physical windows:
      - Min T window: 00:00..09:00 MST (overnight decoupling, radiative cooling, drainage)
      - Max T window: 10:00..18:00 MST (daytime PBL mixing, solar heating, afternoon moisture)
    """
    t0 = time.monotonic()
    df = hourly_df.copy()
    local_time = df["time_utc"] + pd.Timedelta(hours=MST_OFFSET_HOURS)
    df["target_date"] = local_time.dt.floor("D")
    df["hour_mst"] = local_time.dt.hour
    df["dp_spread"] = df["temperature_2m"] - df["dew_point_2m"]

    for optional_col in [
        "snow_depth",
        "boundary_layer_height",
        "cloud_cover_low",
        "wind_gusts_10m",
        "soil_temperature_0_to_7cm",
        "soil_moisture_0_to_7cm",
        "temperature_850hPa",
        "temperature_700hPa",
        "temperature_500hPa",
    ]:
        if optional_col not in df.columns:
            df[optional_col] = np.nan

    df["lapse_700_500"] = df["temperature_700hPa"] - df["temperature_500hPa"]

    min_win = df[(df["hour_mst"] >= 0) & (df["hour_mst"] <= 9)]
    max_win = df[(df["hour_mst"] >= 10) & (df["hour_mst"] <= 18)]

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
        grid_elev_m=("grid_elev_m", "first"),
    )
    g_min["t2m_p10_min_win"] = grp_min["temperature_2m"].quantile(0.10)
    g_min = g_min.reset_index()

    grp_max = max_win.groupby(["station_id", "target_date"], sort=False)
    g_max = grp_max.agg(
        t2m_max_win=("temperature_2m", "max"),
        t2m_mean_max_win=("temperature_2m", "mean"),
        td2m_max_win=("dew_point_2m", "mean"),
        dp_spread_max_win=("dp_spread", "max"),
        wind_max_win=("wind_speed_10m", "max"),
        gust_max_win=("wind_gusts_10m", "max"),
        pblh_max_win=("boundary_layer_height", "max"),
        sw_rad_max_win=("shortwave_radiation", "mean"),
        cloud_max_win=("cloud_cover", "mean"),
        precip_day=("precipitation", "sum"),
        soil_t_max_win=("soil_temperature_0_to_7cm", "mean"),
        soil_m_max_win=("soil_moisture_0_to_7cm", "mean"),
        t850_max_win=("temperature_850hPa", "mean"),
        t700_max_win=("temperature_700hPa", "mean"),
        lapse_700_500_max_win=("lapse_700_500", "mean"),
    )
    g_max["t2m_p90_max_win"] = grp_max["temperature_2m"].quantile(0.90)
    g_max = g_max.reset_index()

    merged = pd.merge(g_min, g_max, on=["station_id", "target_date"], how="inner")
    rename_map = {
        c: f"{prefix}_{c}"
        for c in merged.columns
        if c not in ("station_id", "target_date", "grid_elev_m")
    }
    rename_map["grid_elev_m"] = f"{prefix}_grid_elev_m"
    out = merged.rename(columns=rename_map)
    print(
        f"  [{_ts()}] [windows] Aggregated {prefix.upper():4s} sub-daily windows -> "
        f"{len(out):,} station-days in {time.monotonic() - t0:.1f}s",
        flush=True,
    )
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

        last_tmin = float(g_min["tmin_f"].iloc[-1]) if len(g_min) >= 1 else np.nan
        prev_tmin = float(g_min["tmin_f"].iloc[-2]) if len(g_min) >= 2 else np.nan
        mean7_tmin = float(g_min["tmin_f"].tail(7).mean()) if len(g_min) >= 3 else np.nan

        last_tmax = float(g_max["tmax_f"].iloc[-1]) if len(g_max) >= 1 else np.nan
        prev_tmax = float(g_max["tmax_f"].iloc[-2]) if len(g_max) >= 2 else np.nan
        mean7_tmax = float(g_max["tmax_f"].tail(7).mean()) if len(g_max) >= 3 else np.nan
        last_snowd = float(g_min["snowd"].iloc[-1]) if len(g_min) >= 1 else np.nan

        rows.append(
            {
                "station_id": sid,
                "causal_last_tmin": last_tmin,
                "causal_prev_tmin": prev_tmin,
                "causal_mean7_tmin": mean7_tmin,
                "causal_last_tmax": last_tmax,
                "causal_prev_tmax": prev_tmax,
                "causal_mean7_tmax": mean7_tmax,
                "causal_last_snowd": last_snowd,
            }
        )
    return pd.DataFrame(rows)


def build_aligned_tensors(
    stations_df: pd.DataFrame,
    coop_df: pd.DataFrame,
    hourly_by_source: dict[str, pd.DataFrame],
    issue_hour_utc: int = 12,
    morning_min_gate_utc: int = 15,
    afternoon_max_gate_utc: int = 23,
    min_causal_lag_hours: int = 48,
) -> TensorBundle:
    t_start = time.monotonic()
    daily_sources: dict[str, pd.DataFrame] = {}
    for src_name, hdf in hourly_by_source.items():
        daily_sources[src_name] = _aggregate_daily_windows(hdf, prefix=src_name)

    primary_src = "gfs" if "gfs" in daily_sources else "era5"
    base_daily = daily_sources[primary_src]
    for src_name, ddf in daily_sources.items():
        if src_name == primary_src:
            continue
        base_daily = pd.merge(
            base_daily,
            ddf,
            on=["station_id", "target_date"],
            how="left",
        )

    if "gfs_t2m_min_win" in base_daily.columns and "ifs_t2m_min_win" in base_daily.columns:
        base_daily["ens_spread_tmin"] = (base_daily["gfs_t2m_min_win"] - base_daily["ifs_t2m_min_win"]).abs()
        base_daily["ens_spread_tmax"] = (base_daily["gfs_t2m_max_win"] - base_daily["ifs_t2m_max_win"]).abs()

    grid_elev_cols = [c for c in base_daily.columns if c.endswith("_grid_elev_m")]
    dyn_cols = [
        c for c in base_daily.columns
        if c not in ("station_id", "target_date") and c not in grid_elev_cols
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
    all_dyn_names = dyn_cols + causal_cols

    st_meta = stations_df.set_index("station_id")
    tier_codes = {
        "basin_plateau": [1.0, 0.0, 0.0],
        "rim_slope": [0.0, 1.0, 0.0],
        "canyon": [0.0, 0.0, 1.0],
    }

    coop_by_sid = {
        sid: g.sort_values("obs_date").reset_index(drop=True)
        for sid, g in coop_df.groupby("station_id")
    }
    dyn_by_sid = {
        sid: g.sort_values("target_date").reset_index(drop=True)
        for sid, g in base_daily.groupby("station_id")
    }

    min_lag_days_min = int(np.ceil(max(1, (min_causal_lag_hours + morning_min_gate_utc - issue_hour_utc) / 24.0)))
    min_lag_days_max = int(np.ceil(max(1, (min_causal_lag_hours + afternoon_max_gate_utc - issue_hour_utc) / 24.0)))
    if issue_hour_utc < morning_min_gate_utc:
        min_lag_days_min = max(min_lag_days_min, 1)
    if issue_hour_utc < afternoon_max_gate_utc:
        min_lag_days_max = max(min_lag_days_max, 1)

    hrrr_col_indices = [i for i, name in enumerate(all_dyn_names) if name.startswith("hrrr_")]
    prim_min_col = dyn_cols.index(f"{primary_src}_t2m_min_win")
    prim_max_col = dyn_cols.index(f"{primary_src}_t2m_max_win")
    prim_snow_col = dyn_cols.index(f"{primary_src}_snow_min_win") if f"{primary_src}_snow_min_win" in dyn_cols else -1
    hrrr_min_col = dyn_cols.index("hrrr_t2m_min_win") if "hrrr_t2m_min_win" in dyn_cols else -1
    hrrr_max_col = dyn_cols.index("hrrr_t2m_max_win") if "hrrr_t2m_max_win" in dyn_cols else -1

    LAPSE_F_PER_M = 0.01168

    issue_dates_out = []
    sids_out = []
    tiers_out = []
    x_dyn_list = []
    mask_dyn_list = []
    x_ctx_list = []
    anchor_list = []
    y_list = []
    snow_list = []

    active_sids = [sid for sid in st_meta.index if sid in coop_by_sid and sid in dyn_by_sid]
    total_st = len(active_sids)

    for st_i, sid in enumerate(active_sids, start=1):
        st_t0 = time.monotonic()
        st_row = st_meta.loc[sid]
        elev_m = float(st_row["elev_m"])
        lat = float(st_row["lat"])
        lon = float(st_row["lon"])
        tier_str = str(st_row["tier"])
        tier_oh = tier_codes.get(tier_str, [1.0, 0.0, 0.0])

        st_coop = coop_by_sid[sid]
        st_dyn = dyn_by_sid[sid]

        coop_dates = st_coop["obs_date"].values.astype("datetime64[D]")
        coop_tmin = st_coop["tmin_f"].to_numpy(dtype=np.float32)
        coop_tmax = st_coop["tmax_f"].to_numpy(dtype=np.float32)
        coop_snowd = st_coop["snowd"].to_numpy(dtype=np.float32) if "snowd" in st_coop.columns else np.zeros(len(st_coop), dtype=np.float32)

        dyn_dates = st_dyn["target_date"].values.astype("datetime64[D]")
        dyn_mat = st_dyn[dyn_cols].to_numpy(dtype=np.float32)

        prim_ge_col = f"{primary_src}_grid_elev_m"
        primary_grid_elev = (
            float(st_dyn[prim_ge_col].dropna().iloc[0])
            if prim_ge_col in st_dyn.columns and st_dyn[prim_ge_col].notna().any()
            else elev_m
        )
        hrrr_grid_elev = (
            float(st_dyn["hrrr_grid_elev_m"].dropna().iloc[0])
            if "hrrr_grid_elev_m" in st_dyn.columns and st_dyn["hrrr_grid_elev_m"].notna().any()
            else primary_grid_elev
        )
        elev_delta_m = elev_m - primary_grid_elev

        coop_map = {d: idx for idx, d in enumerate(coop_dates)}
        dyn_map = {d: idx for idx, d in enumerate(dyn_dates)}

        st_samples = 0
        for issue_d64 in coop_dates:
            horizon_d64 = [issue_d64 + np.timedelta64(d, "D") for d in range(5)]
            if not all((hd in coop_map and hd in dyn_map) for hd in horizon_d64):
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
            anchor_sample = np.zeros((5, 2), dtype=np.float32)
            snow_sample = np.zeros(5, dtype=bool)
            valid_sample = True

            for d_idx, hd in enumerate(horizon_d64):
                ci = coop_map[hd]
                di = dyn_map[hd]
                row_vals = dyn_mat[di]

                y_sample[d_idx, 0] = coop_tmin[ci]
                y_sample[d_idx, 1] = coop_tmax[ci]

                dyn_sample[d_idx, : len(dyn_cols)] = row_vals
                dyn_sample[d_idx, len(dyn_cols) :] = c_vals

                if d_idx >= 2 and hrrr_col_indices:
                    dyn_sample[d_idx, hrrr_col_indices] = np.nan

                raw_min = np.nan
                raw_max = np.nan
                if d_idx < 2 and hrrr_min_col >= 0 and not np.isnan(row_vals[hrrr_min_col]):
                    raw_min = float(row_vals[hrrr_min_col]) - LAPSE_F_PER_M * (elev_m - hrrr_grid_elev)
                    raw_max = float(row_vals[hrrr_max_col]) - LAPSE_F_PER_M * (elev_m - hrrr_grid_elev)

                if np.isnan(raw_min):
                    raw_min = float(row_vals[prim_min_col]) - LAPSE_F_PER_M * elev_delta_m
                    raw_max = float(row_vals[prim_max_col]) - LAPSE_F_PER_M * elev_delta_m

                if np.isnan(raw_min) or np.isnan(raw_max):
                    valid_sample = False
                    break

                anchor_sample[d_idx, 0] = raw_min
                anchor_sample[d_idx, 1] = raw_max

                mod_snow = float(row_vals[prim_snow_col]) if prim_snow_col >= 0 and not np.isnan(row_vals[prim_snow_col]) else 0.0
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
                    sin_doy,
                    cos_doy,
                    *tier_oh,
                ],
                dtype=np.float32,
            )

            issue_dates_out.append(issue_d64)
            sids_out.append(sid)
            tiers_out.append(tier_str)
            x_dyn_list.append(dyn_clean)
            mask_dyn_list.append(mask_sample)
            x_ctx_list.append(ctx_vec)
            anchor_list.append(anchor_sample)
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
        "sin_doy",
        "cos_doy",
        "tier_basin",
        "tier_rim",
        "tier_canyon",
    ]

    order = np.argsort(np.array(issue_dates_out))
    print(
        f"  [{_ts()}] [tensors] Total aligned dataset ready in {time.monotonic() - t_start:.1f}s",
        flush=True,
    )
    return TensorBundle(
        issue_dates=np.array(issue_dates_out)[order],
        station_ids=np.array(sids_out)[order],
        tiers=np.array(tiers_out)[order],
        x_dyn=np.stack(x_dyn_list, axis=0)[order],
        mask_dyn=np.stack(mask_dyn_list, axis=0)[order],
        x_ctx=np.stack(x_ctx_list, axis=0)[order],
        raw_anchor=np.stack(anchor_list, axis=0)[order],
        y_true=np.stack(y_list, axis=0)[order],
        snow_flag=np.stack(snow_list, axis=0)[order],
        dyn_feature_names=all_dyn_names,
        ctx_feature_names=ctx_names,
    )
