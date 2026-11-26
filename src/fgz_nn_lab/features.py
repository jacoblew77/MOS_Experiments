from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MST_OFFSET_HOURS = -7  # Arizona stays on MST (UTC-7) all year


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
    Aggregates sub-daily hourly NWP/reanalysis steps into MST physical windows:
      - Min T window: 00:00..09:00 MST (overnight decoupling, radiative cooling, drainage)
      - Max T window: 10:00..18:00 MST (daytime PBL mixing, solar heating, afternoon moisture)
    """
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

    g_min = min_win.groupby(["station_id", "target_date"], as_index=False).agg(
        t2m_min_win=("temperature_2m", "min"),
        t2m_p10_min_win=("temperature_2m", lambda x: np.nanpercentile(x, 10) if x.notna().any() else np.nan),
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

    g_max = max_win.groupby(["station_id", "target_date"], as_index=False).agg(
        t2m_max_win=("temperature_2m", "max"),
        t2m_p90_max_win=("temperature_2m", lambda x: np.nanpercentile(x, 90) if x.notna().any() else np.nan),
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

    merged = pd.merge(g_min, g_max, on=["station_id", "target_date"], how="inner")
    rename_map = {
        c: f"{prefix}_{c}"
        for c in merged.columns
        if c not in ("station_id", "target_date", "grid_elev_m")
    }
    rename_map["grid_elev_m"] = f"{prefix}_grid_elev_m"
    return merged.rename(columns=rename_map)


def compute_causal_obs_lags(
    coop_df: pd.DataFrame,
    issue_date: pd.Timestamp,
    issue_hour_utc: int,
    morning_min_gate_utc: int = 15,
    afternoon_max_gate_utc: int = 23,
    min_causal_lag_hours: int = 48,
) -> pd.DataFrame:
    """
    Computes strictly causal observation lag features per station for a forecast issued at
    (issue_date, issue_hour_utc).
    Respects both explicit UTC gating rules AND min_causal_lag_hours safety floor.
    """
    issue_ts_utc = pd.Timestamp(issue_date) + pd.Timedelta(hours=issue_hour_utc)

    # For any historical obs_date D:
    # Morning Min on date D becomes gated at D + morning_min_gate_utc hours.
    # Afternoon Max on date D becomes gated at D + afternoon_max_gate_utc hours.
    min_ready_cutoff = issue_ts_utc - pd.Timedelta(hours=max(0, min_causal_lag_hours - morning_min_gate_utc))
    max_ready_cutoff = issue_ts_utc - pd.Timedelta(hours=max(0, min_causal_lag_hours - afternoon_max_gate_utc))

    df = coop_df.copy()
    df["min_gate_ts"] = df["obs_date"] + pd.Timedelta(hours=morning_min_gate_utc)
    df["max_gate_ts"] = df["obs_date"] + pd.Timedelta(hours=afternoon_max_gate_utc)

    valid_min = df[(df["min_gate_ts"] <= issue_ts_utc) & (df["min_gate_ts"] <= min_ready_cutoff)]
    valid_max = df[(df["max_gate_ts"] <= issue_ts_utc) & (df["max_gate_ts"] <= max_ready_cutoff)]

    rows = []
    for sid, grp_all in df.groupby("station_id"):
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
    """
    Builds (N, 5, C_dyn) dynamic sequence tensors, explicit masks, static site/DOY context,
    lapse-adjusted baseline anchors, and 5-day COOP targets (d = 1..5, where d=1 is issue_date).
    Enforces HRRR 48-hour horizon masking (masked out for d >= 3).
    """
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

    # Compute multi-model spread when both GFS and IFS are present
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

    coop_indexed = coop_df.set_index(["station_id", "obs_date"]).sort_index()
    dyn_indexed = base_daily.set_index(["station_id", "target_date"]).sort_index()
    st_meta = stations_df.set_index("station_id")

    tier_codes = {"basin_plateau": [1.0, 0.0, 0.0], "rim_slope": [0.0, 1.0, 0.0], "canyon": [0.0, 0.0, 1.0]}

    # Pre-index COOP arrays per station for fast vectorized causal lookup
    coop_by_sid = {sid: g.sort_values("obs_date").reset_index(drop=True) for sid, g in coop_df.groupby("station_id")}

    min_lag_days_min = int(np.ceil(max(1, (min_causal_lag_hours + morning_min_gate_utc - issue_hour_utc) / 24.0)))
    min_lag_days_max = int(np.ceil(max(1, (min_causal_lag_hours + afternoon_max_gate_utc - issue_hour_utc) / 24.0)))
    if issue_hour_utc < morning_min_gate_utc:
        min_lag_days_min = max(min_lag_days_min, 1)
    if issue_hour_utc < afternoon_max_gate_utc:
        min_lag_days_max = max(min_lag_days_max, 1)

    all_dates = sorted(coop_df["obs_date"].unique())

    issue_dates_out = []
    sids_out = []
    tiers_out = []
    x_dyn_list = []
    mask_dyn_list = []
    x_ctx_list = []
    anchor_list = []
    y_list = []
    snow_list = []

    hrrr_col_indices = [i for i, name in enumerate(all_dyn_names) if name.startswith("hrrr_")]

    # Standard environmental lapse adjustment: ~3.56 F per 1000 ft = 0.01168 F / m
    LAPSE_F_PER_M = 0.01168

    for sid in st_meta.index:
        if sid not in coop_by_sid:
            continue
        st_row = st_meta.loc[sid]
        elev_m = float(st_row["elev_m"])
        lat = float(st_row["lat"])
        lon = float(st_row["lon"])
        tier_str = str(st_row["tier"])
        tier_oh = tier_codes.get(tier_str, [1.0, 0.0, 0.0])

        st_coop = coop_by_sid[sid]
        coop_dates = st_coop["obs_date"].values

        # Representative grid elevation for terrain offset
        st_dyn_sub = base_daily[base_daily["station_id"] == sid]
        if st_dyn_sub.empty:
            continue
        primary_grid_elev = float(st_dyn_sub[f"{primary_src}_grid_elev_m"].dropna().iloc[0]) if f"{primary_src}_grid_elev_m" in st_dyn_sub.columns and st_dyn_sub[f"{primary_src}_grid_elev_m"].notna().any() else elev_m
        elev_delta_m = elev_m - primary_grid_elev

        for issue_dt in all_dates:
            issue_ts = pd.Timestamp(issue_dt)
            horizon_dates = [issue_ts + pd.Timedelta(days=d) for d in range(5)]

            # Require complete 5-day COOP targets and base dynamic rows
            valid_sample = True
            y_sample = np.zeros((5, 2), dtype=np.float32)
            dyn_sample = np.full((5, len(all_dyn_names)), np.nan, dtype=np.float32)
            anchor_sample = np.zeros((5, 2), dtype=np.float32)
            snow_sample = np.zeros(5, dtype=bool)

            # Strictly causal lag lookup
            cutoff_min_dt = (issue_ts - pd.Timedelta(days=min_lag_days_min)).to_datetime64()
            cutoff_max_dt = (issue_ts - pd.Timedelta(days=min_lag_days_max)).to_datetime64()

            idx_min = np.searchsorted(coop_dates, cutoff_min_dt, side="right") - 1
            idx_max = np.searchsorted(coop_dates, cutoff_max_dt, side="right") - 1

            c_vals = np.full(len(causal_cols), np.nan, dtype=np.float32)
            if idx_min >= 2:
                c_vals[0] = float(st_coop["tmin_f"].iloc[idx_min])
                c_vals[1] = float(st_coop["tmin_f"].iloc[idx_min - 1])
                c_vals[2] = float(st_coop["tmin_f"].iloc[max(0, idx_min - 6): idx_min + 1].mean())
                c_vals[6] = float(st_coop["snowd"].iloc[idx_min])
            if idx_max >= 2:
                c_vals[3] = float(st_coop["tmax_f"].iloc[idx_max])
                c_vals[4] = float(st_coop["tmax_f"].iloc[idx_max - 1])
                c_vals[5] = float(st_coop["tmax_f"].iloc[max(0, idx_max - 6): idx_max + 1].mean())

            for d_idx, h_dt in enumerate(horizon_dates):
                if (sid, h_dt) not in coop_indexed.index or (sid, h_dt) not in dyn_indexed.index:
                    valid_sample = False
                    break
                obs_r = coop_indexed.loc[(sid, h_dt)]
                dyn_r = dyn_indexed.loc[(sid, h_dt)]

                y_sample[d_idx, 0] = float(obs_r["tmin_f"])
                y_sample[d_idx, 1] = float(obs_r["tmax_f"])

                row_vals = dyn_r[dyn_cols].to_numpy(dtype=np.float32)
                dyn_sample[d_idx, : len(dyn_cols)] = row_vals
                dyn_sample[d_idx, len(dyn_cols) :] = c_vals

                # Enforce HRRR 48h horizon cutoff: Days 3, 4, 5 (d_idx >= 2) are masked out
                if d_idx >= 2 and hrrr_col_indices:
                    dyn_sample[d_idx, hrrr_col_indices] = np.nan

                # Raw lapse-adjusted anchor temperature (prefer HRRR on Days 1-2 if valid, else primary)
                raw_min = np.nan
                raw_max = np.nan
                if d_idx < 2 and f"hrrr_t2m_min_win" in dyn_r.index and pd.notna(dyn_r["hrrr_t2m_min_win"]):
                    hrrr_ge = float(dyn_r["hrrr_grid_elev_m"]) if pd.notna(dyn_r.get("hrrr_grid_elev_m", np.nan)) else primary_grid_elev
                    raw_min = float(dyn_r["hrrr_t2m_min_win"]) - LAPSE_F_PER_M * (elev_m - hrrr_ge)
                    raw_max = float(dyn_r["hrrr_t2m_max_win"]) - LAPSE_F_PER_M * (elev_m - hrrr_ge)

                if np.isnan(raw_min):
                    raw_min = float(dyn_r[f"{primary_src}_t2m_min_win"]) - LAPSE_F_PER_M * elev_delta_m
                    raw_max = float(dyn_r[f"{primary_src}_t2m_max_win"]) - LAPSE_F_PER_M * elev_delta_m

                if np.isnan(raw_min) or np.isnan(raw_max):
                    valid_sample = False
                    break

                anchor_sample[d_idx, 0] = raw_min
                anchor_sample[d_idx, 1] = raw_max

                snow_col = f"{primary_src}_snow_min_win"
                obs_snow = float(obs_r["snowd"]) if "snowd" in obs_r.index else 0.0
                mod_snow = float(dyn_r[snow_col]) if snow_col in dyn_r.index and pd.notna(dyn_r[snow_col]) else 0.0
                snow_sample[d_idx] = (obs_snow > 0.1) or (mod_snow > 0.01)

            if not valid_sample:
                continue

            mask_sample = (~np.isnan(dyn_sample)).astype(np.float32)
            dyn_clean = np.nan_to_num(dyn_sample, nan=0.0)

            doy = issue_ts.dayofyear
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

            issue_dates_out.append(np.datetime64(issue_ts.strftime("%Y-%m-%d")))
            sids_out.append(sid)
            tiers_out.append(tier_str)
            x_dyn_list.append(dyn_clean)
            mask_dyn_list.append(mask_sample)
            x_ctx_list.append(ctx_vec)
            anchor_list.append(anchor_sample)
            y_list.append(y_sample)
            snow_list.append(snow_sample)

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