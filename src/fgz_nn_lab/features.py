from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MST_OFFSET_HOURS = -7
LAPSE_F_PER_M = 0.01168

PRIMARY_SLOTS = [
    "prim_t2m_min_win",
    "prim_t2m_p10_min_win",
    "prim_t2m_mean_min_win",
    "prim_t2m_min_morn24",
    "prim_t2m_min_eve24",
    "prim_td2m_min_win",
    "prim_dp_spread_min_win",
    "prim_wind_min_win",
    "prim_wind_med_min_win",
    "prim_pblh_min_win",
    "prim_cloud_min_win",
    "prim_snow_min_win",
    "prim_t850_min_win",
    "prim_t700_min_win",
    "prim_decouple_idx",
    "prim_t2m_max_cal",
    "prim_t2m_p90_max_cal",
    "prim_t2m_mean_max_cal",
    "prim_t2m_max_morn24",
    "prim_t2m_max_eve24",
    "prim_td2m_max_cal",
    "prim_dp_spread_max_cal",
    "prim_wind_max_cal",
    "prim_pblh_max_cal",
    "prim_sw_rad_max_cal",
    "prim_cloud_max_cal",
    "prim_precip_day",
    "prim_soil_t_max_cal",
    "prim_soil_m_max_cal",
    "prim_t850_max_cal",
    "prim_t700_max_cal",
]

IFS_SLOTS = [
    "ifs_t2m_min_win",
    "ifs_td2m_min_win",
    "ifs_dp_spread_min_win",
    "ifs_wind_min_win",
    "ifs_cloud_min_win",
    "ifs_t2m_max_cal",
    "ifs_t2m_max_morn24",
    "ifs_td2m_max_cal",
    "ifs_wind_max_cal",
    "ifs_sw_rad_max_cal",
    "ifs_cloud_max_cal",
    "ifs_precip_day",
]

NBM_SLOTS = [
    "nbm_t2m_min_win",
    "nbm_td2m_min_win",
    "nbm_wind_min_win",
    "nbm_cloud_min_win",
    "nbm_t2m_max_cal",
    "nbm_t2m_max_morn24",
    "nbm_td2m_max_cal",
    "nbm_wind_max_cal",
    "nbm_sw_rad_max_cal",
    "nbm_precip_day",
]

HRRR_SLOTS = [
    "hrrr_t2m_min_win",
    "hrrr_t2m_p10_min_win",
    "hrrr_td2m_min_win",
    "hrrr_dp_spread_min_win",
    "hrrr_wind_min_win",
    "hrrr_cloud_min_win",
    "hrrr_decouple_idx",
    "hrrr_t2m_max_cal",
    "hrrr_t2m_max_morn24",
    "hrrr_td2m_max_cal",
    "hrrr_wind_max_cal",
    "hrrr_sw_rad_max_cal",
    "hrrr_cloud_max_cal",
]

MOS_AND_CAUSAL_SLOTS = [
    "mos_bulletin_tmin",
    "mos_bulletin_tmax",
    "mos_bulletin_dpt",
    "mos_bulletin_wsp",
    "ens_spread_tmin",
    "ens_spread_tmax_cal",
    "nbm_minus_gfs_tmin",
    "nbm_minus_gfs_tmax",
    "causal_last_tmin",
    "causal_prev_tmin",
    "causal_mean7_tmin",
    "causal_last_tmax",
    "causal_prev_tmax",
    "causal_mean7_tmax",
    "causal_last_snowd",
]

CANONICAL_DYN_FEATURES = PRIMARY_SLOTS + IFS_SLOTS + NBM_SLOTS + HRRR_SLOTS + MOS_AND_CAUSAL_SLOTS


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


@dataclass
class TensorBundle:
    issue_dates: np.ndarray
    station_ids: np.ndarray
    networks: np.ndarray
    station_indices: np.ndarray
    station_vocab: list[str]
    tiers: np.ndarray
    x_dyn: np.ndarray
    mask_dyn: np.ndarray
    x_ctx: np.ndarray
    raw_anchor: np.ndarray
    raw_anchor_candidates: np.ndarray
    y_true: np.ndarray
    snow_flag: np.ndarray
    dyn_feature_names: list[str]
    ctx_feature_names: list[str]


def _aggregate_station_windows(
    hourly_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    t0 = time.monotonic()
    obs_hr_map = stations_df.set_index("station_id")["obs_hour_mst"].to_dict()

    df = hourly_df.copy()
    local_time = df["time_utc"] + pd.Timedelta(hours=MST_OFFSET_HOURS)
    df["target_date"] = local_time.dt.floor("D")
    df["hour_mst"] = local_time.dt.hour

    st_obs_hr = df["station_id"].map(obs_hr_map).fillna(7).astype(int)
    morn_hr = np.where((st_obs_hr >= 5) & (st_obs_hr <= 10), st_obs_hr, 7)
    df["morn24_date"] = (local_time + pd.to_timedelta(24 - morn_hr, unit="h")).dt.floor("D")
    df["eve24_date"] = (local_time + pd.Timedelta(hours=7)).dt.floor("D")

    for col in [
        "dew_point_2m", "precipitation", "snow_depth", "cloud_cover",
        "shortwave_radiation", "wind_speed_10m", "boundary_layer_height",
        "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
        "temperature_850hPa", "temperature_700hPa",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["dp_spread"] = df["temperature_2m"] - df["dew_point_2m"]
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
        pblh_max_cal=("boundary_layer_height", "max"),
        sw_rad_max_cal=("shortwave_radiation", "mean"),
        cloud_max_cal=("cloud_cover", "mean"),
        precip_day=("precipitation", "sum"),
        soil_t_max_cal=("soil_temperature_0_to_7cm", "mean"),
        soil_m_max_cal=("soil_moisture_0_to_7cm", "mean"),
        t850_max_cal=("temperature_850hPa", "mean"),
        t700_max_cal=("temperature_700hPa", "mean"),
    )
    g_max["t2m_p90_max_cal"] = grp_max["temperature_2m"].quantile(0.90)
    g_max = g_max.reset_index()

    g_morn = (
        df.groupby(["station_id", "morn24_date"], sort=False)
        .agg(t2m_max_morn24=("temperature_2m", "max"), t2m_min_morn24=("temperature_2m", "min"))
        .reset_index()
        .rename(columns={"morn24_date": "target_date"})
    )
    g_eve = (
        df.groupby(["station_id", "eve24_date"], sort=False)
        .agg(t2m_max_eve24=("temperature_2m", "max"), t2m_min_eve24=("temperature_2m", "min"))
        .reset_index()
        .rename(columns={"eve24_date": "target_date"})
    )

    merged = pd.merge(g_min, g_max, on=["station_id", "target_date"], how="inner")
    merged = pd.merge(merged, g_morn, on=["station_id", "target_date"], how="left")
    merged = pd.merge(merged, g_eve, on=["station_id", "target_date"], how="left")

    print(f"  [{_ts()}] [windows] Aggregated {label:12s} -> {len(merged):,} station-days in {time.monotonic() - t0:.1f}s", flush=True)
    return merged


def _extract_prev_run_lead_df(prev_df: pd.DataFrame, lead_idx: int) -> pd.DataFrame:
    suffix = "" if lead_idx == 0 else f"_previous_day{lead_idx}"
    out = pd.DataFrame(
        {
            "station_id": prev_df["station_id"],
            "time_utc": prev_df["time_utc"],
            "grid_elev_m": prev_df["grid_elev_m"],
        }
    )
    for bv in [
        "temperature_2m", "dew_point_2m", "precipitation", "cloud_cover",
        "shortwave_radiation", "wind_speed_10m", "temperature_850hPa", "temperature_700hPa",
    ]:
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
    nbm_df: pd.DataFrame | None,
    hrrr_df: pd.DataFrame | None,
) -> pd.DataFrame:
    base = prim_df.copy()
    if era5_df is not None and era5_df is not prim_df:
        era5_sub = era5_df[
            ["station_id", "target_date", "soil_t_max_cal", "soil_m_max_cal", "snow_min_win", "pblh_min_win", "pblh_max_cal"]
        ].rename(
            columns={
                "soil_t_max_cal": "_e_soil_t",
                "soil_m_max_cal": "_e_soil_m",
                "snow_min_win": "_e_snow",
                "pblh_min_win": "_e_pbl_min",
                "pblh_max_cal": "_e_pbl_max",
            }
        )
        base = pd.merge(base, era5_sub, on=["station_id", "target_date"], how="left")
        base["soil_t_max_cal"] = base["soil_t_max_cal"].fillna(base["_e_soil_t"])
        base["soil_m_max_cal"] = base["soil_m_max_cal"].fillna(base["_e_soil_m"])
        base["snow_min_win"] = base["snow_min_win"].fillna(base["_e_snow"])
        base["pblh_min_win"] = base["pblh_min_win"].fillna(base["_e_pbl_min"])
        base["pblh_max_cal"] = base["pblh_max_cal"].fillna(base["_e_pbl_max"])

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

    for prefix, src_df, slots, keep_ge in [
        ("ifs_", ifs_df, IFS_SLOTS, False),
        ("nbm_", nbm_df, NBM_SLOTS, True),
        ("hrrr_", hrrr_df, HRRR_SLOTS, True),
    ]:
        ge_col = f"{prefix}grid_elev_m"
        if src_df is not None:
            ren = {s.replace(prefix, "", 1): s for s in slots}
            cols = ["station_id", "target_date"] + (["grid_elev_m"] if keep_ge else []) + [k for k in ren if k in src_df.columns]
            sub = src_df[cols].rename(columns={**ren, **({"grid_elev_m": ge_col} if keep_ge else {})})
            out = pd.merge(out, sub, on=["station_id", "target_date"], how="left")
        elif keep_ge:
            out[ge_col] = np.nan
        for s in slots:
            if s not in out.columns:
                out[s] = np.nan

    out["ens_spread_tmin"] = (out["prim_t2m_min_win"] - out["ifs_t2m_min_win"]).abs() if ifs_df is not None else np.nan
    out["ens_spread_tmax_cal"] = (out["prim_t2m_max_cal"] - out["ifs_t2m_max_cal"]).abs() if ifs_df is not None else np.nan
    out["nbm_minus_gfs_tmin"] = (out["nbm_t2m_min_win"] - out["prim_t2m_min_win"]) if nbm_df is not None else np.nan
    out["nbm_minus_gfs_tmax"] = (out["nbm_t2m_max_cal"] - out["prim_t2m_max_cal"]) if nbm_df is not None else np.nan
    return out


def build_aligned_tensors(
    stations_df: pd.DataFrame,
    coop_df: pd.DataFrame,
    hourly_by_source: dict[str, pd.DataFrame],
    mos_df: pd.DataFrame | None = None,
    issue_hour_utc: int = 12,
    morning_min_gate_utc: int = 15,
    afternoon_max_gate_utc: int = 23,
    min_causal_lag_hours: int = 48,
    station_vocab: list[str] | None = None,
) -> TensorBundle:
    t_start = time.monotonic()
    st_df = stations_df.copy()
    if "network" not in st_df.columns:
        st_df["network"] = "AZ_COOP"
    if "mos_id" not in st_df.columns:
        st_df["mos_id"] = "KFLG"

    has_prev_runs = "prev_gfs" in hourly_by_source
    era5_daily = _aggregate_station_windows(hourly_by_source["era5"], st_df, "ERA5") if "era5" in hourly_by_source else None
    gfs_hist_daily = _aggregate_station_windows(hourly_by_source["gfs_hist"], st_df, "GFS_HIST") if "gfs_hist" in hourly_by_source else (
        _aggregate_station_windows(hourly_by_source["gfs"], st_df, "GFS") if "gfs" in hourly_by_source else None
    )
    ifs_direct = _aggregate_station_windows(hourly_by_source["ifs"], st_df, "IFS") if "ifs" in hourly_by_source else None
    nbm_direct = _aggregate_station_windows(hourly_by_source["nbm"], st_df, "NBM") if "nbm" in hourly_by_source else None
    hrrr_hist_daily = _aggregate_station_windows(hourly_by_source["hrrr"], st_df, "HRRR") if "hrrr" in hourly_by_source else None

    lead_tables: list[pd.DataFrame] = []
    if has_prev_runs:
        for d_idx in range(5):
            pgfs_d = _aggregate_station_windows(
                _extract_prev_run_lead_df(hourly_by_source["prev_gfs"], d_idx), st_df, f"PREV_GFS_D{d_idx + 1}"
            )
            if gfs_hist_daily is not None and d_idx == 0:
                extra_cols = ["station_id", "target_date", "pblh_min_win", "pblh_max_cal", "snow_min_win", "t850_min_win", "t700_min_win", "t850_max_cal", "t700_max_cal"]
                avail_extra = [c for c in extra_cols if c in gfs_hist_daily.columns]
                sub_g = gfs_hist_daily[avail_extra]
                pgfs_d = pd.merge(pgfs_d, sub_g, on=["station_id", "target_date"], how="left", suffixes=("", "_ghist"))
                for c in avail_extra:
                    if f"{c}_ghist" in pgfs_d.columns:
                        pgfs_d[c] = pgfs_d[c].fillna(pgfs_d[f"{c}_ghist"])
                        pgfs_d = pgfs_d.drop(columns=[f"{c}_ghist"])

            pifs_d = _aggregate_station_windows(_extract_prev_run_lead_df(hourly_by_source["prev_ifs"], d_idx), st_df, f"PREV_IFS_D{d_idx + 1}") if "prev_ifs" in hourly_by_source else None
            pnbm_d = _aggregate_station_windows(_extract_prev_run_lead_df(hourly_by_source["prev_nbm"], d_idx), st_df, f"PREV_NBM_D{d_idx + 1}") if "prev_nbm" in hourly_by_source else None
            phrrr_d = _aggregate_station_windows(_extract_prev_run_lead_df(hourly_by_source["prev_hrrr"], d_idx), st_df, f"PREV_HRRR_D{d_idx + 1}") if (d_idx < 2 and "prev_hrrr" in hourly_by_source) else None

            lead_tables.append(
                _merge_canonical_day_table(
                    prim_df=pgfs_d,
                    era5_df=era5_daily if d_idx == 0 else None,
                    ifs_df=pifs_d,
                    nbm_df=pnbm_d,
                    hrrr_df=phrrr_d,
                )
            )
    else:
        prim_base = gfs_hist_daily if gfs_hist_daily is not None else era5_daily
        assert prim_base is not None
        shared_tbl = _merge_canonical_day_table(prim_base, era5_daily, ifs_direct, nbm_direct, hrrr_hist_daily)
        lead_tables = [shared_tbl for _ in range(5)]

    table_cols = PRIMARY_SLOTS + IFS_SLOTS + NBM_SLOTS + HRRR_SLOTS + [
        "ens_spread_tmin", "ens_spread_tmax_cal", "nbm_minus_gfs_tmin", "nbm_minus_gfs_tmax"
    ]
    mos_cols = ["mos_bulletin_tmin", "mos_bulletin_tmax", "mos_bulletin_dpt", "mos_bulletin_wsp"]
    causal_cols = [
        "causal_last_tmin", "causal_prev_tmin", "causal_mean7_tmin",
        "causal_last_tmax", "causal_prev_tmax", "causal_mean7_tmax", "causal_last_snowd",
    ]
    all_dyn_names = PRIMARY_SLOTS + IFS_SLOTS + NBM_SLOTS + HRRR_SLOTS + MOS_AND_CAUSAL_SLOTS
    assert all_dyn_names == CANONICAL_DYN_FEATURES

    mos_lookup: dict[tuple[str, np.datetime64, np.datetime64], tuple[float, float, float, float]] = {}
    if mos_df is not None and not mos_df.empty:
        i_d64 = pd.to_datetime(mos_df["issue_date"]).values.astype("datetime64[D]")
        t_d64 = pd.to_datetime(mos_df["target_date"]).values.astype("datetime64[D]")
        for m_id, idt, tdt, mmin, mmax, mdpt, mwsp in zip(
            mos_df["mos_id"].astype(str), i_d64, t_d64,
            mos_df["mos_tmin_f"], mos_df["mos_tmax_f"], mos_df["mos_dpt_mean"], mos_df["mos_wsp_mean"]
        ):
            mos_lookup[(m_id, idt, tdt)] = (float(mmin), float(mmax), float(mdpt), float(mwsp))

    st_meta = st_df.set_index("station_id")
    vocab = station_vocab if station_vocab is not None else sorted(st_df["station_id"].tolist())
    sid_to_idx = {sid: i for i, sid in enumerate(vocab)}
    tier_codes = {"basin_plateau": [1.0, 0.0, 0.0], "rim_slope": [0.0, 1.0, 0.0], "canyon": [0.0, 0.0, 1.0]}

    coop_by_sid = {sid: g.sort_values("obs_date").reset_index(drop=True) for sid, g in coop_df.groupby("station_id")}
    lead_by_sid = [
        {sid: g.sort_values("target_date").reset_index(drop=True) for sid, g in tbl.groupby("station_id")}
        for tbl in lead_tables
    ]

    min_lag_days_min = int(np.ceil(max(1, (min_causal_lag_hours + morning_min_gate_utc - issue_hour_utc) / 24.0)))
    min_lag_days_max = int(np.ceil(max(1, (min_causal_lag_hours + afternoon_max_gate_utc - issue_hour_utc) / 24.0)))

    hrrr_col_indices = [i for i, name in enumerate(all_dyn_names) if name.startswith("hrrr_")]
    pos_map = {name: i for i, name in enumerate(all_dyn_names)}
    tbl_dest_indices = [pos_map[c] for c in table_cols]
    mos_dest_indices = [pos_map[c] for c in mos_cols]
    causal_dest_indices = [pos_map[c] for c in causal_cols]

    ix_p_min_cal = table_cols.index("prim_t2m_min_win")
    ix_p_min_m24 = table_cols.index("prim_t2m_min_morn24")
    ix_p_min_e24 = table_cols.index("prim_t2m_min_eve24")
    ix_p_max_cal = table_cols.index("prim_t2m_max_cal")
    ix_p_max_m24 = table_cols.index("prim_t2m_max_morn24")
    ix_p_max_e24 = table_cols.index("prim_t2m_max_eve24")
    ix_p_snow = table_cols.index("prim_snow_min_win")

    ix_n_min = table_cols.index("nbm_t2m_min_win")
    ix_n_max_cal = table_cols.index("nbm_t2m_max_cal")
    ix_n_max_m24 = table_cols.index("nbm_t2m_max_morn24")

    ix_h_min = table_cols.index("hrrr_t2m_min_win")
    ix_h_max_cal = table_cols.index("hrrr_t2m_max_cal")

    issue_dates_out, sids_out, nets_out, sidx_out, tiers_out = [], [], [], [], []
    x_dyn_list, mask_dyn_list, x_ctx_list = [], [], []
    anc_def_list, anc_cand_list, y_list, snow_list = [], [], [], []

    active_sids = [sid for sid in st_meta.index if sid in coop_by_sid and sid in lead_by_sid[0]]

    for st_i, sid in enumerate(active_sids, start=1):
        st_row = st_meta.loc[sid]
        elev_m = float(st_row["elev_m"])
        lat, lon = float(st_row["lat"]), float(st_row["lon"])
        obs_hr = float(st_row["obs_hour_mst"])
        net_str = str(st_row["network"])
        is_asos = 1.0 if net_str == "AZ_ASOS" else 0.0
        mos_id = str(st_row["mos_id"])
        tier_str = str(st_row["tier"])
        tier_oh = tier_codes.get(tier_str, [1.0, 0.0, 0.0])
        st_int_id = int(sid_to_idx.get(sid, 0))

        st_coop = coop_by_sid[sid]
        coop_dates = st_coop["obs_date"].values.astype("datetime64[D]")
        coop_tmin = st_coop["tmin_f"].to_numpy(dtype=np.float32)
        coop_tmax = st_coop["tmax_f"].to_numpy(dtype=np.float32)
        coop_snowd = st_coop["snowd"].to_numpy(dtype=np.float32)
        coop_map = {d: idx for idx, d in enumerate(coop_dates)}

        d_maps, d_mats = [], []
        prim_ge, nbm_ge, hrrr_ge = elev_m, elev_m, elev_m
        for d_idx in range(5):
            sdf = lead_by_sid[d_idx].get(sid)
            if sdf is None or sdf.empty:
                d_maps.append({})
                d_mats.append(np.empty((0, len(table_cols)), dtype=np.float32))
                continue
            if d_idx == 0:
                if sdf["prim_grid_elev_m"].notna().any():
                    prim_ge = float(sdf["prim_grid_elev_m"].dropna().iloc[0])
                nbm_ge = float(sdf["nbm_grid_elev_m"].dropna().iloc[0]) if sdf["nbm_grid_elev_m"].notna().any() else prim_ge
                hrrr_ge = float(sdf["hrrr_grid_elev_m"].dropna().iloc[0]) if sdf["hrrr_grid_elev_m"].notna().any() else prim_ge
            dd = sdf["target_date"].values.astype("datetime64[D]")
            d_maps.append({dt_val: i for i, dt_val in enumerate(dd)})
            d_mats.append(sdf[table_cols].to_numpy(dtype=np.float32))

        p_delta = elev_m - prim_ge
        n_delta = elev_m - nbm_ge
        h_delta = elev_m - hrrr_ge

        st_samples = 0
        for issue_d64 in coop_dates:
            horizon_d64 = [issue_d64 + np.timedelta64(d, "D") for d in range(5)]
            if not all((hd in coop_map and hd in d_maps[d_i]) for d_i, hd in enumerate(horizon_d64)):
                continue

            idx_min = int(np.searchsorted(coop_dates, issue_d64 - np.timedelta64(min_lag_days_min, "D"), side="right")) - 1
            idx_max = int(np.searchsorted(coop_dates, issue_d64 - np.timedelta64(min_lag_days_max, "D"), side="right")) - 1

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
            anc_cand = np.zeros((5, 6), dtype=np.float32)
            snow_sample = np.zeros(5, dtype=bool)
            valid_sample = True

            for d_idx, hd in enumerate(horizon_d64):
                ci = coop_map[hd]
                di = d_maps[d_idx][hd]
                rv = d_mats[d_idx][di]

                y_sample[d_idx, 0] = coop_tmin[ci]
                y_sample[d_idx, 1] = coop_tmax[ci]

                dyn_sample[d_idx, tbl_dest_indices] = rv
                dyn_sample[d_idx, causal_dest_indices] = c_vals

                m_tup = mos_lookup.get((mos_id, issue_d64, hd))
                if m_tup is not None:
                    dyn_sample[d_idx, mos_dest_indices] = np.array(m_tup, dtype=np.float32)

                if d_idx >= 2 and hrrr_col_indices:
                    dyn_sample[d_idx, hrrr_col_indices] = np.nan

                p_min_c = float(rv[ix_p_min_cal]) - LAPSE_F_PER_M * p_delta
                p_min_m = (float(rv[ix_p_min_m24]) - LAPSE_F_PER_M * p_delta) if not np.isnan(rv[ix_p_min_m24]) else p_min_c
                p_min_e = (float(rv[ix_p_min_e24]) - LAPSE_F_PER_M * p_delta) if not np.isnan(rv[ix_p_min_e24]) else p_min_c

                p_max_c = float(rv[ix_p_max_cal]) - LAPSE_F_PER_M * p_delta
                p_max_m = (float(rv[ix_p_max_m24]) - LAPSE_F_PER_M * p_delta) if not np.isnan(rv[ix_p_max_m24]) else p_max_c
                p_max_e = (float(rv[ix_p_max_e24]) - LAPSE_F_PER_M * p_delta) if not np.isnan(rv[ix_p_max_e24]) else p_max_c

                if np.isnan(p_min_c) or np.isnan(p_max_c):
                    valid_sample = False
                    break

                if not np.isnan(rv[ix_n_min]) and not np.isnan(rv[ix_n_max_cal]):
                    n_min_c = float(rv[ix_n_min]) - LAPSE_F_PER_M * n_delta
                    n_max_c = float(rv[ix_n_max_cal]) - LAPSE_F_PER_M * n_delta
                    n_max_m = (float(rv[ix_n_max_m24]) - LAPSE_F_PER_M * n_delta) if not np.isnan(rv[ix_n_max_m24]) else n_max_c
                    p_min_c = 0.55 * n_min_c + 0.45 * p_min_c
                    p_max_c = 0.55 * n_max_c + 0.45 * p_max_c
                    p_max_m = 0.55 * n_max_m + 0.45 * p_max_m
                    p_max_e = 0.55 * n_max_c + 0.45 * p_max_e
                elif d_idx < 2 and not np.isnan(rv[ix_h_min]) and not np.isnan(rv[ix_h_max_cal]):
                    h_min_c = float(rv[ix_h_min]) - LAPSE_F_PER_M * h_delta
                    h_max_c = float(rv[ix_h_max_cal]) - LAPSE_F_PER_M * h_delta
                    p_min_c = 0.50 * h_min_c + 0.50 * p_min_c
                    p_max_c = 0.50 * h_max_c + 0.50 * p_max_c

                anc_cand[d_idx] = [p_min_c, p_min_m, p_min_e, p_max_c, p_max_m, p_max_e]
                mod_snow = float(rv[ix_p_snow]) if not np.isnan(rv[ix_p_snow]) else 0.0
                snow_sample[d_idx] = (coop_snowd[ci] > 0.1) or (mod_snow > 0.01)

            if not valid_sample:
                continue

            mask_sample = (~np.isnan(dyn_sample)).astype(np.float32)
            dyn_clean = np.nan_to_num(dyn_sample, nan=0.0)

            doy = pd.Timestamp(issue_d64).day_of_year
            ctx_vec = np.array(
                [
                    elev_m / 1000.0,
                    p_delta / 1000.0,
                    (lat - 35.2),
                    (lon + 111.5),
                    obs_hr / 24.0,
                    is_asos,
                    float(np.sin(2.0 * np.pi * doy / 365.25)),
                    float(np.cos(2.0 * np.pi * doy / 365.25)),
                    *tier_oh,
                ],
                dtype=np.float32,
            )

            issue_dates_out.append(issue_d64)
            sids_out.append(sid)
            nets_out.append(net_str)
            sidx_out.append(st_int_id)
            tiers_out.append(tier_str)
            x_dyn_list.append(dyn_clean)
            mask_dyn_list.append(mask_sample)
            x_ctx_list.append(ctx_vec)
            anc_def_list.append(np.stack([anc_cand[:, 0], anc_cand[:, 3]], axis=-1))
            anc_cand_list.append(anc_cand)
            y_list.append(y_sample)
            snow_list.append(snow_sample)
            st_samples += 1

        print(f"  [{_ts()}] [tensors {st_i:02d}/{len(active_sids):02d}] {sid:6s} ({net_str}): {st_samples:,} sequences", flush=True)

    ctx_names = [
        "elev_km", "elev_delta_km", "lat_centered", "lon_centered", "obs_hour_norm",
        "is_asos", "sin_doy", "cos_doy", "tier_basin", "tier_rim", "tier_canyon",
    ]
    order = np.argsort(np.array(issue_dates_out))
    print(f"  [{_ts()}] [tensors] Ready ({len(order):,} samples, {len(all_dyn_names)} channels) in {time.monotonic() - t_start:.1f}s", flush=True)
    return TensorBundle(
        issue_dates=np.array(issue_dates_out)[order],
        station_ids=np.array(sids_out)[order],
        networks=np.array(nets_out)[order],
        station_indices=np.array(sidx_out, dtype=np.int64)[order],
        station_vocab=vocab,
        tiers=np.array(tiers_out)[order],
        x_dyn=np.stack(x_dyn_list, axis=0)[order],
        mask_dyn=np.stack(mask_dyn_list, axis=0)[order],
        x_ctx=np.stack(x_ctx_list, axis=0)[order],
        raw_anchor=np.stack(anc_def_list, axis=0)[order],
        raw_anchor_candidates=np.stack(anc_cand_list, axis=0)[order],
        y_true=np.stack(y_list, axis=0)[order],
        snow_flag=np.stack(snow_list, axis=0)[order],
        dyn_feature_names=all_dyn_names,
        ctx_feature_names=ctx_names,
    )


def build_live_inference_bundle(
    stations_df: pd.DataFrame,
    coop_df: pd.DataFrame,
    live_hourly: dict[str, pd.DataFrame],
    live_mos_df: pd.DataFrame,
    station_vocab: list[str],
    issue_hour_utc: int = 12,
    morning_min_gate_utc: int = 15,
    afternoon_max_gate_utc: int = 23,
    min_causal_lag_hours: int = 48,
) -> tuple[TensorBundle | None, list[str]]:
    """
    Builds today's live 5-day operational TensorBundle across all FGZ stations, passing
    gfs + ifs + nbm + hrrr directly into build_aligned_tensors so NBM anchor blending &
    IFS ensemble spread match training 1:1.
    """
    if "gfs" not in live_hourly:
        return None, []

    today_mst = (pd.Timestamp(datetime.now(timezone.utc)) + pd.Timedelta(hours=MST_OFFSET_HOURS)).floor("D").tz_localize(None)
    horizon_dates = [today_mst + pd.Timedelta(days=d) for d in range(5)]

    synth_obs_rows = []
    for sid in stations_df["station_id"]:
        for hd in horizon_dates + [today_mst - pd.Timedelta(days=k) for k in range(1, 10)]:
            synth_obs_rows.append({"station_id": sid, "obs_date": hd, "tmin_f": 40.0, "tmax_f": 65.0, "snowd": 0.0})
    synth_df = pd.DataFrame(synth_obs_rows)

    if coop_df is not None and not coop_df.empty:
        real_sub = coop_df[coop_df["obs_date"] < today_mst][["station_id", "obs_date", "tmin_f", "tmax_f", "snowd"]]
        comb_obs = pd.concat([real_sub, synth_df], ignore_index=True).drop_duplicates(subset=["station_id", "obs_date"], keep="first")
    else:
        comb_obs = synth_df

    bundle = build_aligned_tensors(
        stations_df=stations_df,
        coop_df=comb_obs,
        hourly_by_source=live_hourly,
        mos_df=live_mos_df,
        issue_hour_utc=issue_hour_utc,
        morning_min_gate_utc=morning_min_gate_utc,
        afternoon_max_gate_utc=afternoon_max_gate_utc,
        min_causal_lag_hours=min_causal_lag_hours,
        station_vocab=station_vocab,
    )

    today_d64 = today_mst.to_datetime64().astype("datetime64[D]")
    m_today = bundle.issue_dates == today_d64
    if not m_today.any():
        return None, [d.strftime("%Y-%m-%d") for d in horizon_dates]

    idx_sel = np.where(m_today)[0]
    live_b = TensorBundle(
        issue_dates=bundle.issue_dates[idx_sel],
        station_ids=bundle.station_ids[idx_sel],
        networks=bundle.networks[idx_sel],
        station_indices=bundle.station_indices[idx_sel],
        station_vocab=station_vocab,
        tiers=bundle.tiers[idx_sel],
        x_dyn=bundle.x_dyn[idx_sel],
        mask_dyn=bundle.mask_dyn[idx_sel],
        x_ctx=bundle.x_ctx[idx_sel],
        raw_anchor=bundle.raw_anchor[idx_sel],
        raw_anchor_candidates=bundle.raw_anchor_candidates[idx_sel],
        y_true=bundle.y_true[idx_sel],
        snow_flag=bundle.snow_flag[idx_sel],
        dyn_feature_names=bundle.dyn_feature_names,
        ctx_feature_names=bundle.ctx_feature_names,
    )
    return live_b, [d.strftime("%Y-%m-%d") for d in horizon_dates]