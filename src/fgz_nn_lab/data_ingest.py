from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

FT_TO_M = 0.3048
IEM_GEOJSON_URL = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
IEM_DAILY_API_TXT_URL = "https://mesonet.agron.iastate.edu/api/1/daily.txt"
IEM_MOS_CGI_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
IEM_MOS_LIVE_URL = "https://mesonet.agron.iastate.edu/api/1/mos.json"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

ERA5_HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "snow_depth",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "boundary_layer_height",
]

HIST_SUPPLEMENT_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "snow_depth",
    "boundary_layer_height",
    "temperature_850hPa",
    "temperature_700hPa",
    "temperature_500hPa",
]

PREV_SURFACE_BASE_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
]

PREV_UPPER_BASE_VARS = [
    "temperature_850hPa",
    "temperature_700hPa",
]

LIVE_FORECAST_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "snow_depth",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "boundary_layer_height",
    "temperature_850hPa",
    "temperature_700hPa",
    "temperature_500hPa",
]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def _classify_fgz_tier(sid: str, name: str, elev_ft: float) -> str:
    up = f"{sid} {name}".upper()
    if any(k in up for k in ("PHANTOM", "PHRA3", "PAGE", "PGA", "CANYON", "LEES FERRY", "SUPAI")):
        if elev_ft < 4600.0:
            return "canyon"
    if any(k in up for k in ("RIM", "PAYSON", "PRESCOTT", "PRC", "SHOW LOW", "SOW", "GRAND CANYON", "GCN", "JEROME", "PINE")):
        return "rim_slope"
    return "basin_plateau"


def _build_prev_runs_var_list(max_prev_day: int, include_upper: bool = False) -> list[str]:
    base_list = list(PREV_SURFACE_BASE_VARS)
    if include_upper:
        base_list.extend(PREV_UPPER_BASE_VARS)
    out: list[str] = []
    for bv in base_list:
        out.append(bv)
        for k in range(1, max_prev_day + 1):
            out.append(f"{bv}_previous_day{k}")
    return out


def _nearest_asos_mos_id(lat: float, lon: float, asos_rows: list[dict[str, Any]]) -> str:
    if not asos_rows:
        return "KFLG"
    best_id = "KFLG"
    best_d2 = 1e12
    for r in asos_rows:
        d2 = (lat - float(r["lat"])) ** 2 + (lon - float(r["lon"])) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_id = str(r["mos_id"])
    return best_id


def load_station_catalog(config_path: Path, raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Loads curated FGZ seed stations AND auto-discovers active WFO Flagstaff (wfo == 'FGZ')
    AZ_COOP and AZ_ASOS stations directly from IEM GeoJSON metadata.
    """
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = payload.get("elevation_overrides_ft", {})
    excluded = {str(x).upper() for x in payload.get("exclude_stations", ["AZ0000"])}
    auto_disc = bool(payload.get("auto_discover_fgz", False))

    seed_list = payload.get("seed_stations", payload.get("stations", []))
    by_key: dict[str, dict[str, Any]] = {}

    for st in seed_list:
        sid = str(st["station_id"]).strip().upper()
        if sid in excluded or sid.startswith("AZC"):
            continue
        net = str(st.get("network", "AZ_COOP")).strip().upper()
        elev_ft = float(overrides.get(sid, st["elev_ft"]))
        mos_id = str(st.get("mos_id", f"K{sid}" if net == "AZ_ASOS" and len(sid) == 3 else "KFLG")).upper()
        by_key[f"{net}:{sid}"] = {
            "station_id": sid,
            "network": net,
            "mos_id": mos_id,
            "name": str(st["name"]),
            "lat": float(st["lat"]),
            "lon": float(st["lon"]),
            "elev_ft": elev_ft,
            "elev_m": elev_ft * FT_TO_M,
            "tier": str(st.get("tier", _classify_fgz_tier(sid, str(st["name"]), elev_ft))),
            "obs_hour_mst": int(st.get("obs_hour_mst", 0 if net == "AZ_ASOS" else 7)),
            "is_seed": True,
        }

    if auto_disc and raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        meta_cache = raw_dir / "iem_fgz_station_discovery_v3.json"
        discovered: list[dict[str, Any]] = []
        if meta_cache.exists():
            discovered = json.loads(meta_cache.read_text(encoding="utf-8"))
        else:
            for net in ("AZ_ASOS", "AZ_COOP"):
                try:
                    r = requests.get(IEM_GEOJSON_URL.format(network=net), timeout=45)
                    if r.status_code == 200:
                        feats = r.json().get("features", [])
                        for f in feats:
                            props = f.get("properties", {})
                            geom = f.get("geometry", {})
                            coords = geom.get("coordinates", [None, None])
                            wfo = str(props.get("wfo", "")).strip().upper()
                            online = bool(props.get("online", True))
                            archive_end = props.get("archive_end")
                            sid = str(props.get("sid", f.get("id", ""))).strip().upper()
                            if wfo != "FGZ" or not online or archive_end not in (None, "", "null"):
                                continue
                            if not sid or sid in excluded or sid.startswith("AZC") or coords[0] is None:
                                continue
                            elev_m_raw = float(props.get("elevation", 1800.0) or 1800.0)
                            elev_ft = float(overrides.get(sid, elev_m_raw / FT_TO_M))
                            discovered.append(
                                {
                                    "station_id": sid,
                                    "network": net,
                                    "name": str(props.get("sname", sid)),
                                    "lat": float(coords[1]),
                                    "lon": float(coords[0]),
                                    "elev_ft": elev_ft,
                                }
                            )
                except requests.RequestException:
                    continue
            if discovered:
                meta_cache.write_text(json.dumps(discovered, indent=2), encoding="utf-8")

        asos_seeds = [v for v in by_key.values() if v["network"] == "AZ_ASOS"]
        for d in discovered:
            k = f"{d['network']}:{d['station_id']}"
            if k in by_key:
                continue
            sid = d["station_id"]
            net = d["network"]
            elev_ft = float(overrides.get(sid, d["elev_ft"]))
            mos_id = f"K{sid}" if net == "AZ_ASOS" and len(sid) == 3 else _nearest_asos_mos_id(
                float(d["lat"]), float(d["lon"]), asos_seeds
            )
            by_key[k] = {
                "station_id": sid,
                "network": net,
                "mos_id": mos_id,
                "name": d["name"],
                "lat": float(d["lat"]),
                "lon": float(d["lon"]),
                "elev_ft": elev_ft,
                "elev_m": elev_ft * FT_TO_M,
                "tier": _classify_fgz_tier(sid, d["name"], elev_ft),
                "obs_hour_mst": 0 if net == "AZ_ASOS" else 7,
                "is_seed": False,
            }

    df = pd.DataFrame(list(by_key.values()))
    df["uid"] = np.where(df["network"] == "AZ_ASOS", "K" + df["station_id"], df["station_id"])
    return df.reset_index(drop=True)


def filter_coop_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces strict real-observation QC across both daily.py (max_temp_f/min_temp_f) and legacy schemas:
      - Rejects synthetic/climate-division IDs (AZ0000, AZC*)
      - Rejects any row flagged estimated (temp_estimated, high_estimated, low_estimated)
      - Rejects blanks, NaNs, sentinel codes (-99, 999), and physical crossings (low > high)
    """
    out = df.copy()
    st_col = "station" if "station" in out.columns else "station_id"
    out[st_col] = out[st_col].astype(str).str.strip().str.upper()
    out = out[(out[st_col] != "AZ0000") & (~out[st_col].str.startswith("AZC"))].copy()

    for est_col in ("temp_estimated", "high_estimated", "low_estimated"):
        if est_col in out.columns:
            keep_mask = (
                out[est_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["false", "f", "0", "no", "n", "", "none", "nan"])
            )
            # If explicit 'true' / 't' / '1' / 'yes' / 'y' appears, reject it strictly
            bad_mask = (
                out[est_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "t", "1", "yes", "y"])
            )
            out = out[keep_mask & (~bad_mask)].copy()

    hi_col = "max_temp_f" if "max_temp_f" in out.columns else "high"
    lo_col = "min_temp_f" if "min_temp_f" in out.columns else "low"

    out[hi_col] = pd.to_numeric(out[hi_col], errors="coerce")
    out[lo_col] = pd.to_numeric(out[lo_col], errors="coerce")
    out = out.dropna(subset=[hi_col, lo_col]).copy()
    out = out[(out[hi_col] >= -40.0) & (out[hi_col] <= 132.0)].copy()
    out = out[(out[lo_col] >= -55.0) & (out[lo_col] <= 112.0)].copy()
    out = out[out[lo_col] <= out[hi_col]].copy()
    return out


def fetch_unestimated_iem_observations(
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    min_clean_days: int = 380,
    min_recent_days: int = 150,
    max_coop: int = 12,
    max_asos: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pulls 100% genuine, un-imputed station observations via IEM's /cgi-bin/request/daily.py
    (with na=blank, never Climodat spatial estimates), verifies recent completeness, and caps
    to the cleanest active FGZ COOP + ASOS roster.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = f"{start_date}_{end_date}_{min_clean_days}d_{len(stations_df)}cand"
    obs_cache = raw_dir / f"iem_real_daily_v3_{cache_tag}.parquet"
    aud_cache = raw_dir / f"iem_real_audit_v3_{cache_tag}.parquet"
    sel_cache = raw_dir / f"iem_selected_stations_v3_{cache_tag}.parquet"

    if obs_cache.exists() and aud_cache.exists() and sel_cache.exists():
        obs_df = pd.read_parquet(obs_cache)
        aud_df = pd.read_parquet(aud_cache)
        sel_df = pd.read_parquet(sel_cache)
        print(
            f"  [{_ts()}] [OBS] Cache HIT -> {len(obs_df):,} genuine un-estimated rows across "
            f"{len(sel_df)} active FGZ stations ({ (sel_df['network']=='AZ_ASOS').sum() } ASOS, "
            f"{ (sel_df['network']=='AZ_COOP').sum() } COOP)",
            flush=True,
        )
        return obs_df, aud_df, sel_df

    dt_start = pd.Timestamp(start_date)
    dt_end = pd.Timestamp(end_date)
    recent_cutoff = dt_end - pd.Timedelta(days=365)

    net_frames: list[pd.DataFrame] = []
    raw_count_by_uid: dict[str, int] = {}

    for net_i, (net_name, net_grp) in enumerate(stations_df.groupby("network")):
        if net_i > 0:
            time.sleep(2.5)  # Respect IEM daily.py documented 2-second sequential throttle
        sids = net_grp["station_id"].tolist()
        uid_map = dict(zip(net_grp["station_id"], net_grp["uid"]))
        print(
            f"  [{_ts()}] [OBS] Querying IEM daily.py (na=blank, non-estimated) for {net_name} "
            f"({len(sids)} candidate stations, {start_date}..{end_date})...",
            flush=True,
        )
        params = {
            "network": net_name,
            "stations": ",".join(sids),
            "year1": dt_start.year,
            "month1": dt_start.month,
            "day1": dt_start.day,
            "year2": dt_end.year,
            "month2": dt_end.month,
            "day2": dt_end.day,
            "var": "max_temp_f,min_temp_f,precip_in,snow_in,snowd_in",
            "na": "blank",
            "format": "csv",
        }
        resp = requests.get(IEM_DAILY_URL, params=params, timeout=120)
        resp.raise_for_status()

        lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if len(lines) <= 1:
            continue
        ndf = pd.read_csv(io.StringIO("\n".join(lines)))
        ndf.columns = [c.strip().lower() for c in ndf.columns]
        ndf["station"] = ndf["station"].astype(str).str.strip().str.upper()
        ndf["uid"] = ndf["station"].map(uid_map)
        ndf = ndf.dropna(subset=["uid"]).copy()

        for u, cnt in ndf.groupby("uid")["day"].count().items():
            raw_count_by_uid[str(u)] = int(cnt)

        qc_n = filter_coop_qc(ndf)
        qc_n["obs_date"] = pd.to_datetime(qc_n["day"]).dt.tz_localize(None)
        qc_n = qc_n.rename(
            columns={
                "max_temp_f": "tmax_f",
                "min_temp_f": "tmin_f",
                "precip_in": "precip",
                "snow_in": "snow",
                "snowd_in": "snowd",
            }
        )
        for col in ("precip", "snow", "snowd"):
            qc_n[col] = pd.to_numeric(qc_n.get(col, 0.0), errors="coerce").fillna(0.0)
        qc_n["network"] = net_name
        net_frames.append(qc_n[["uid", "station", "network", "obs_date", "tmax_f", "tmin_f", "precip", "snow", "snowd"]])

    all_qc = pd.concat(net_frames, ignore_index=True) if net_frames else pd.DataFrame()

    # Cross-check recent AZ_COOP daily.txt for explicit temp_estimated flags & observer reporting hour
    coop_est_bad: set[tuple[str, str]] = set()
    detected_obs_hr: dict[str, int] = {}
    if not all_qc.empty and (stations_df["network"] == "AZ_COOP").any():
        for yr in range(max(dt_start.year, dt_end.year - 2), dt_end.year + 1):
            for st_r in stations_df[stations_df["network"] == "AZ_COOP"].itertuples():
                try:
                    r_txt = requests.get(
                        IEM_DAILY_API_TXT_URL,
                        params={"network": "AZ_COOP", "station": st_r.station_id, "year": yr},
                        timeout=25,
                    )
                    if r_txt.status_code == 200 and "," in r_txt.text:
                        t_lines = [x for x in r_txt.text.splitlines() if x.strip() and not x.startswith("#")]
                        if len(t_lines) > 1:
                            tdf = pd.read_csv(io.StringIO("\n".join(t_lines)))
                            tdf.columns = [c.strip().lower() for c in tdf.columns]
                            if "temp_estimated" in tdf.columns and "date" in tdf.columns:
                                bad_rows = tdf[
                                    tdf["temp_estimated"].astype(str).str.strip().str.lower().isin(["true", "t", "1", "yes"])
                                ]
                                for bd in bad_rows["date"].astype(str):
                                    coop_est_bad.add((str(st_r.uid), bd[:10]))
                            for hr_col in ("temp_hour", "obs_hour", "hour"):
                                if hr_col in tdf.columns:
                                    hrs = pd.to_numeric(tdf[hr_col], errors="coerce").dropna()
                                    if not hrs.empty:
                                        detected_obs_hr[str(st_r.uid)] = int(hrs.mode().iloc[0])
                except Exception:
                    continue

    if coop_est_bad and not all_qc.empty:
        date_strs = all_qc["obs_date"].dt.strftime("%Y-%m-%d")
        drop_est = [
            (u, d) in coop_est_bad for u, d in zip(all_qc["uid"].astype(str), date_strs)
        ]
        n_dropped_est = int(sum(drop_est))
        if n_dropped_est > 0:
            print(f"  [{_ts()}] [OBS QC] Purged {n_dropped_est} rows flagged temp_estimated=True in daily.txt", flush=True)
            all_qc = all_qc[~np.array(drop_est)].copy()

    qc_counts = all_qc.groupby("uid")["obs_date"].count().to_dict() if not all_qc.empty else {}
    rec_counts = (
        all_qc[all_qc["obs_date"] >= recent_cutoff].groupby("uid")["obs_date"].count().to_dict()
        if not all_qc.empty
        else {}
    )

    st_work = stations_df.copy()
    for uid_k, hr_v in detected_obs_hr.items():
        st_work.loc[st_work["uid"] == uid_k, "obs_hour_mst"] = hr_v

    st_work["raw_days"] = st_work["uid"].map(lambda u: int(raw_count_by_uid.get(u, 0)))
    st_work["qc_clean_days"] = st_work["uid"].map(lambda u: int(qc_counts.get(u, 0)))
    st_work["recent_365d_clean"] = st_work["uid"].map(lambda u: int(rec_counts.get(u, 0)))
    st_work["eligible"] = (st_work["qc_clean_days"] >= min_clean_days) & (
        st_work["recent_365d_clean"] >= min_recent_days
    )

    # Rank eligible stations: prioritize seed stations first, then highest clean observation count
    elig_asos = (
        st_work[(st_work["network"] == "AZ_ASOS") & st_work["eligible"]]
        .sort_values(["is_seed", "qc_clean_days"], ascending=[False, False])
        .head(max_asos)
    )
    elig_coop = (
        st_work[(st_work["network"] == "AZ_COOP") & st_work["eligible"]]
        .sort_values(["is_seed", "qc_clean_days"], ascending=[False, False])
        .head(max_coop)
    )
    kept_uids = set(elig_asos["uid"]).union(set(elig_coop["uid"]))

    audit_rows = []
    for _, r in st_work.iterrows():
        u = str(r["uid"])
        kept = u in kept_uids
        if kept:
            reason = "active_kept"
        elif r["qc_clean_days"] < min_clean_days:
            reason = f"below_min_clean_days({r['qc_clean_days']}<{min_clean_days})"
        elif r["recent_365d_clean"] < min_recent_days:
            reason = f"sparse_recent_year({r['recent_365d_clean']}<{min_recent_days})"
        else:
            reason = "ranked_beyond_cap"
        audit_rows.append(
            {
                "station_id": u,
                "iem_sid": r["station_id"],
                "network": r["network"],
                "mos_id": r["mos_id"],
                "name": r["name"],
                "elev_ft": float(r["elev_ft"]),
                "tier": r["tier"],
                "obs_hour_mst": int(r["obs_hour_mst"]),
                "raw_days": int(r["raw_days"]),
                "qc_clean_days": int(r["qc_clean_days"]),
                "recent_365d_clean": int(r["recent_365d_clean"]),
                "kept": kept,
                "status_reason": reason,
            }
        )
    audit_df = pd.DataFrame(audit_rows)

    selected_df = st_work[st_work["uid"].isin(kept_uids)].copy()
    selected_df["station_id"] = selected_df["uid"]
    selected_df = selected_df.sort_values(["network", "station_id"]).reset_index(drop=True)

    obs_final = (
        all_qc[all_qc["uid"].isin(kept_uids)]
        .drop(columns=["station"])
        .rename(columns={"uid": "station_id"})
        .sort_values(["station_id", "obs_date"])
        .reset_index(drop=True)
    )

    obs_final.to_parquet(obs_cache, index=False)
    audit_df.to_parquet(aud_cache, index=False)
    selected_df.to_parquet(sel_cache, index=False)

    print(
        f"  [{_ts()}] [OBS] Locked {len(obs_final):,} 100%-real observations across "
        f"{len(selected_df)} active FGZ sites ({len(elig_asos)} ASOS + {len(elig_coop)} COOP)",
        flush=True,
    )
    return obs_final, audit_df, selected_df


def fetch_iem_mos_bulletins(
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Fetches historical 12Z GFS MOS bulletins from IEM (/cgi-bin/request/mos.py) for all unique
    FGZ ASOS MOS stations (KFLG, KPRC, KINW, KSJN, KGCN, KPGA, KRQE, KSOW) and extracts
    daily Day 1..4 MOS Min/Max temperature guidance per issue_date & target_date.
    Gracefully returns an empty DataFrame if mos.py is unavailable for a station.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    mos_ids = sorted(stations_df["mos_id"].dropna().unique().tolist())
    mos_cache = raw_dir / f"iem_gfs_mos_v3_{start_date}_{end_date}_{len(mos_ids)}st.parquet"
    if mos_cache.exists():
        mdf = pd.read_parquet(mos_cache)
        print(f"  [{_ts()}] [MOS Bulletins] Cache HIT -> {mos_cache.name} ({len(mdf):,} rows)", flush=True)
        return mdf

    dt_s = pd.Timestamp(start_date)
    dt_e = pd.Timestamp(end_date) + pd.Timedelta(days=5)
    frames: list[pd.DataFrame] = []

    print(
        f"  [{_ts()}] [MOS Bulletins] Fetching IEM GFS MOS archive for {len(mos_ids)} FGZ airports ({mos_ids})...",
        flush=True,
    )
    for m_id in mos_ids:
        try:
            params = {
                "station": m_id,
                "model": "GFS",
                "sts": dt_s.strftime("%Y-%m-%dT00:00Z"),
                "ets": dt_e.strftime("%Y-%m-%dT23:59Z"),
                "format": "csv",
            }
            r = requests.get(IEM_MOS_CGI_URL, params=params, timeout=75)
            if r.status_code != 200 or "," not in r.text:
                time.sleep(1.5)
                continue
            lines = [ln for ln in r.text.splitlines() if ln.strip() and not ln.startswith("#")]
            if len(lines) <= 1:
                time.sleep(1.5)
                continue
            df = pd.read_csv(io.StringIO("\n".join(lines)), low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]
            if "runtime" not in df.columns or "ftime" not in df.columns:
                time.sleep(1.5)
                continue

            df["runtime_utc"] = pd.to_datetime(df["runtime"], utc=True, errors="coerce").dt.tz_localize(None)
            df["ftime_utc"] = pd.to_datetime(df["ftime"], utc=True, errors="coerce").dt.tz_localize(None)
            df = df.dropna(subset=["runtime_utc", "ftime_utc"]).copy()

            # Keep 12Z operational cycle (matches our 12:00Z daily issue time!)
            df = df[df["runtime_utc"].dt.hour == 12].copy()
            if df.empty:
                time.sleep(1.5)
                continue

            df["issue_date"] = df["runtime_utc"].dt.floor("D")
            local_ftime = df["ftime_utc"] - pd.Timedelta(hours=7)
            df["target_date"] = local_ftime.dt.floor("D")
            df["tmp_num"] = pd.to_numeric(df.get("tmp", np.nan), errors="coerce")
            df["nx_num"] = pd.to_numeric(df.get("n_x", np.nan), errors="coerce")
            df["dpt_num"] = pd.to_numeric(df.get("dpt", np.nan), errors="coerce")
            df["wsp_num"] = pd.to_numeric(df.get("wsp", np.nan), errors="coerce")

            # Aggregate 3-hourly tmp and explicit n_x extremum per (issue_date, target_date)
            grp = df.groupby(["issue_date", "target_date"], as_index=False).agg(
                mos_tmin_3h=("tmp_num", "min"),
                mos_tmax_3h=("tmp_num", "max"),
                mos_nx_min=("nx_num", "min"),
                mos_nx_max=("nx_num", "max"),
                mos_dpt_mean=("dpt_num", "mean"),
                mos_wsp_mean=("wsp_num", "mean"),
            )
            grp["mos_tmin_f"] = grp["mos_nx_min"].combine_first(grp["mos_tmin_3h"])
            grp["mos_tmax_f"] = grp["mos_nx_max"].combine_first(grp["mos_tmax_3h"])
            grp["mos_id"] = m_id
            frames.append(grp[["mos_id", "issue_date", "target_date", "mos_tmin_f", "mos_tmax_f", "mos_dpt_mean", "mos_wsp_mean"]])
            print(f"    [{_ts()}] [MOS] {m_id}: extracted {len(grp):,} daily 12Z GFS-MOS projections", flush=True)
            time.sleep(2.0)
        except Exception as exc:
            print(f"    [{_ts()}] [MOS] {m_id} skipped ({type(exc).__name__})", flush=True)
            time.sleep(1.5)

    full_mos = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["mos_id", "issue_date", "target_date", "mos_tmin_f", "mos_tmax_f", "mos_dpt_mean", "mos_wsp_mean"])
    )
    full_mos.to_parquet(mos_cache, index=False)
    return full_mos


def _request_open_meteo_json(
    session: requests.Session,
    src_label: str,
    url: str,
    params: dict[str, Any],
    retries: int = 24,
    read_timeout: int = 150,
) -> Any:
    """
    Bulletproof single-run Open-Meteo requester: if the rolling 60-minute IP quota fills (HTTP 429),
    it waits patiently (up to ~60 minutes across retries) rather than crashing the workflow.
    """
    delay = 8.0
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=(20, read_timeout))
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 and attempt < retries - 1:
                hdr_wait = float(resp.headers.get("Retry-After", 0.0) or 0.0)
                # Escalate cleanly to 150s sleeps so 24 retries span >55 mins (clearing any 1h rolling window!)
                wait_s = max(hdr_wait, min(delay, 150.0))
                print(
                    f"    [{_ts()}] [{src_label}] Open-Meteo 429 quota pause; waiting {wait_s:.0f}s "
                    f"(attempt {attempt + 1}/{retries}) so single workflow completes cleanly...",
                    flush=True,
                )
                time.sleep(wait_s)
                delay = min(delay * 1.6, 150.0)
                continue
            if resp.status_code in (500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(delay, 45.0))
                delay = min(delay * 1.5, 60.0)
                continue
            resp.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt < retries - 1:
                print(f"    [{_ts()}] [{src_label}] {type(exc).__name__}; retrying in {min(delay, 45.0):.0f}s...", flush=True)
                time.sleep(min(delay, 45.0))
                delay = min(delay * 1.5, 60.0)
                continue
            raise
    raise RuntimeError(f"Open-Meteo request failed after {retries} attempts for {url}")


def _build_date_windows(start_date: str, end_date: str, chunk_months: int) -> list[tuple[str, str]]:
    dt_start = pd.Timestamp(start_date)
    dt_end = pd.Timestamp(end_date)
    windows: list[tuple[str, str]] = []
    cur = dt_start
    while cur <= dt_end:
        chunk_end = min(cur + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1), dt_end)
        windows.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + pd.Timedelta(days=1)
    return windows


def _fetch_batched_multi_station_chunks(
    session: requests.Session,
    src_label: str,
    url: str,
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    requested_vars: list[str],
    canonical_vars: list[str],
    chunk_dir: Path,
    extra_params: dict[str, Any] | None = None,
    chunk_months: int = 6,
    sleep_sec: float = 1.0,
    fallback_vars: list[str] | None = None,
) -> pd.DataFrame:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    windows = _build_date_windows(start_date, end_date, chunk_months)
    total_chunks = len(windows)

    sids = stations_df["station_id"].tolist()
    st_sig = abs(hash("_".join(sids))) % 100000
    lats_csv = ",".join(f"{float(x):.4f}" for x in stations_df["lat"])
    lons_csv = ",".join(f"{float(x):.4f}" for x in stations_df["lon"])
    elev_csv = ",".join(["nan"] * len(sids))

    chunk_frames: list[pd.DataFrame] = []
    src_t0 = time.monotonic()

    for idx, (s_str, e_str) in enumerate(windows, start=1):
        chunk_file = chunk_dir / f"chunk_v3_{src_label}_{s_str}_{e_str}_{len(sids)}st_{st_sig}.parquet"
        if chunk_file.exists():
            print(
                f"  [{_ts()}] [{src_label.upper():9s} {idx:02d}/{total_chunks:02d}] Cache HIT  {s_str} -> {e_str}",
                flush=True,
            )
            chunk_frames.append(pd.read_parquet(chunk_file))
            continue

        print(
            f"  [{_ts()}] [{src_label.upper():9s} {idx:02d}/{total_chunks:02d}] "
            f"Requesting {s_str} -> {e_str} ({len(sids)} stations, {len(requested_vars)} vars)...",
            flush=True,
        )
        t0 = time.monotonic()

        active_vars = list(requested_vars)
        params: dict[str, Any] = {
            "latitude": lats_csv,
            "longitude": lons_csv,
            "start_date": s_str,
            "end_date": e_str,
            "hourly": ",".join(active_vars),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
            "elevation": elev_csv,
        }
        if extra_params:
            params.update(extra_params)

        fb = fallback_vars or ["temperature_2m", "dew_point_2m", "wind_speed_10m", "cloud_cover"]
        try:
            payload = _request_open_meteo_json(session, src_label, url, params)
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 400 and active_vars != fb:
                print(
                    f"    [{_ts()}] [{src_label.upper()}] HTTP 400 on full var list for {s_str}..{e_str}; using core vars...",
                    flush=True,
                )
                active_vars = list(fb)
                params["hourly"] = ",".join(active_vars)
                payload = _request_open_meteo_json(session, src_label, url, params)
            else:
                raise

        items = payload if isinstance(payload, list) else [payload]
        loc_dfs: list[pd.DataFrame] = []
        for sid, item in zip(sids, items):
            hourly = item.get("hourly", {})
            grid_elev = float(item["elevation"]) if item.get("elevation") is not None else np.nan
            cdf = pd.DataFrame({"time_utc": pd.to_datetime(hourly["time"])})
            for v in canonical_vars:
                if v in hourly:
                    cdf[v] = pd.to_numeric(hourly[v], errors="coerce").astype(np.float32)
                else:
                    cdf[v] = np.float32(np.nan)
            cdf["grid_elev_m"] = np.float32(grid_elev)
            cdf["station_id"] = sid
            loc_dfs.append(cdf)

        chunk_df = pd.concat(loc_dfs, ignore_index=True)
        chunk_df.to_parquet(chunk_file, index=False)
        chunk_frames.append(chunk_df)
        print(
            f"    [{_ts()}] [{src_label.upper():9s} {idx:02d}/{total_chunks:02d}] "
            f"Saved {len(chunk_df):,} rows in {time.monotonic() - t0:.1f}s (elapsed: {time.monotonic() - src_t0:.0f}s)",
            flush=True,
        )
        time.sleep(sleep_sec)

    return (
        pd.concat(chunk_frames, ignore_index=True)
        .drop_duplicates(subset=["station_id", "time_utc"])
        .sort_values(["station_id", "time_utc"])
        .reset_index(drop=True)
    )


def fetch_hourly_predictors_for_stations(
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    source_mode: str = "true_lead_nwp",
) -> dict[str, pd.DataFrame]:
    """
    Fetches hourly predictors including NOAA's 2.5 km Gridded MOS/Blend (ncep_nbm_conus)
    and upper-air (850/700 hPa) _previous_day1..4 runs on GFS.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = raw_dir / "chunks_v3"
    pad_start = (pd.Timestamp(start_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    pad_end = (pd.Timestamp(end_date) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    if source_mode == "era5":
        sources = ["era5"]
    else:
        # Operational True-Lead Suite: ERA5 (Day-0 soil/snow), GFS_HIST (Day-0 upper/PBL),
        # PREV_HRRR (Days 1-2 3km), PREV_GFS (Days 1-5 + 850/700hPa), PREV_IFS (Days 1-5), PREV_NBM (2.5km Gridded MOS!)
        sources = ["era5", "gfs_hist", "prev_hrrr", "prev_gfs", "prev_ifs", "prev_nbm"]

    st_sig = abs(hash("_".join(sorted(stations_df["station_id"].tolist())))) % 100000
    collected: dict[str, pd.DataFrame] = {}

    with requests.Session() as session:
        session.headers.update({"Accept-Encoding": "gzip, deflate", "User-Agent": "fgz-nn-lab/0.5"})
        for s_i, src in enumerate(sources, start=1):
            # NBM archive starts 2024-10-03 on Open-Meteo
            eff_start = max(pad_start, "2024-10-05") if src == "prev_nbm" else pad_start
            if pd.Timestamp(eff_start) > pd.Timestamp(pad_end):
                continue

            src_cache = raw_dir / f"hourly_v3_{src}_{eff_start}_{pad_end}_{len(stations_df)}st_{st_sig}.parquet"
            print(f"  [{_ts()}] === Source [{s_i}/{len(sources)}]: {src.upper()} ({eff_start} -> {pad_end}) ===", flush=True)
            if src_cache.exists():
                cdf = pd.read_parquet(src_cache)
                print(f"  [{_ts()}] [{src.upper()}] Cache HIT -> {src_cache.name} ({len(cdf):,} rows)", flush=True)
                collected[src] = cdf
                continue

            if src == "era5":
                df_out = _fetch_batched_multi_station_chunks(
                    session, "era5", OPEN_METEO_ARCHIVE_URL, stations_df,
                    eff_start, pad_end, ERA5_HOURLY_VARS, ERA5_HOURLY_VARS,
                    chunk_dir, chunk_months=12, sleep_sec=0.8,
                )
            elif src == "gfs_hist":
                df_out = _fetch_batched_multi_station_chunks(
                    session, "gfs_hist", OPEN_METEO_HIST_FORECAST_URL, stations_df,
                    eff_start, pad_end, HIST_SUPPLEMENT_VARS, HIST_SUPPLEMENT_VARS,
                    chunk_dir, extra_params={"models": "gfs_seamless"}, chunk_months=8, sleep_sec=0.9,
                )
            elif src == "prev_hrrr":
                v_req = _build_prev_runs_var_list(1, include_upper=False)
                df_out = _fetch_batched_multi_station_chunks(
                    session, "prev_hrrr", OPEN_METEO_PREV_RUNS_URL, stations_df,
                    eff_start, pad_end, v_req, v_req,
                    chunk_dir, extra_params={"models": "ncep_hrrr_conus"}, chunk_months=6, sleep_sec=1.0,
                )
            elif src == "prev_gfs":
                v_full = _build_prev_runs_var_list(4, include_upper=True)
                v_surf = _build_prev_runs_var_list(4, include_upper=False)
                df_out = _fetch_batched_multi_station_chunks(
                    session, "prev_gfs", OPEN_METEO_PREV_RUNS_URL, stations_df,
                    eff_start, pad_end, v_full, v_full,
                    chunk_dir, extra_params={"models": "gfs_seamless"}, chunk_months=4, sleep_sec=1.1,
                    fallback_vars=v_surf,
                )
            elif src == "prev_ifs":
                v_surf = _build_prev_runs_var_list(4, include_upper=False)
                df_out = _fetch_batched_multi_station_chunks(
                    session, "prev_ifs", OPEN_METEO_PREV_RUNS_URL, stations_df,
                    eff_start, pad_end, v_surf, v_surf,
                    chunk_dir, extra_params={"models": "ecmwf_ifs"}, chunk_months=6, sleep_sec=1.0,
                )
            elif src == "prev_nbm":
                v_surf = _build_prev_runs_var_list(4, include_upper=False)
                df_out = _fetch_batched_multi_station_chunks(
                    session, "prev_nbm", OPEN_METEO_PREV_RUNS_URL, stations_df,
                    eff_start, pad_end, v_surf, v_surf,
                    chunk_dir, extra_params={"models": "ncep_nbm_conus"}, chunk_months=6, sleep_sec=1.0,
                )
            else:
                raise ValueError(f"Unknown source: {src}")

            df_out.to_parquet(src_cache, index=False)
            collected[src] = df_out

    return collected


def fetch_live_operational_inputs(
    stations_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Fetches TODAY'S live 6-day operational forecast streams (GFS, HRRR, ECMWF IFS, and 2.5km NBM)
    from api.open-meteo.com/v1/forecast PLUS live IEM GFS/NBS MOS bulletins for operational output.
    """
    sids = stations_df["station_id"].tolist()
    lats_csv = ",".join(f"{float(x):.4f}" for x in stations_df["lat"])
    lons_csv = ",".join(f"{float(x):.4f}" for x in stations_df["lon"])
    elev_csv = ",".join(["nan"] * len(sids))

    live_models = {
        "gfs": "gfs_seamless",
        "ifs": "ecmwf_ifs",
        "hrrr": "ncep_hrrr_conus",
        "nbm": "ncep_nbm_conus",
    }
    live_hourly: dict[str, pd.DataFrame] = {}

    with requests.Session() as session:
        for key, mod_code in live_models.items():
            req_vars = LIVE_FORECAST_VARS if key == "gfs" else [
                "temperature_2m", "dew_point_2m", "precipitation", "cloud_cover", "shortwave_radiation", "wind_speed_10m"
            ]
            params = {
                "latitude": lats_csv,
                "longitude": lons_csv,
                "forecast_days": 7,
                "past_days": 2,
                "hourly": ",".join(req_vars),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "UTC",
                "elevation": elev_csv,
                "models": mod_code,
            }
            try:
                payload = _request_open_meteo_json(session, f"live_{key}", OPEN_METEO_LIVE_FORECAST_URL, params, retries=4)
                items = payload if isinstance(payload, list) else [payload]
                dfs = []
                for sid, item in zip(sids, items):
                    h = item.get("hourly", {})
                    ge = float(item["elevation"]) if item.get("elevation") is not None else np.nan
                    cdf = pd.DataFrame({"time_utc": pd.to_datetime(h["time"])})
                    for v in LIVE_FORECAST_VARS:
                        cdf[v] = pd.to_numeric(h[v], errors="coerce").astype(np.float32) if v in h else np.float32(np.nan)
                    cdf["grid_elev_m"] = np.float32(ge)
                    cdf["station_id"] = sid
                    dfs.append(cdf)
                live_hourly[key] = pd.concat(dfs, ignore_index=True)
            except Exception as exc:
                print(f"    [{_ts()}] [Live {key}] skipped ({type(exc).__name__})", flush=True)

    # Fetch live GFS MOS JSON per unique ASOS MOS ID
    mos_rows: list[dict[str, Any]] = []
    today_utc = pd.Timestamp(datetime.now(timezone.utc).date())
    for m_id in sorted(stations_df["mos_id"].dropna().unique().tolist()):
        try:
            r = requests.get(IEM_MOS_LIVE_URL, params={"station": m_id, "model": "GFS"}, timeout=20)
            if r.status_code == 200:
                entries = r.json().get("data", [])
                if entries:
                    mdf = pd.DataFrame(entries)
                    mdf["ftime_utc"] = pd.to_datetime(mdf["ftime"], utc=True, errors="coerce").dt.tz_localize(None)
                    mdf["target_date"] = (mdf["ftime_utc"] - pd.Timedelta(hours=7)).dt.floor("D")
                    mdf["tmp_num"] = pd.to_numeric(mdf.get("tmp", np.nan), errors="coerce")
                    mdf["nx_num"] = pd.to_numeric(mdf.get("n_x", np.nan), errors="coerce")
                    mdf["dpt_num"] = pd.to_numeric(mdf.get("dpt", np.nan), errors="coerce")
                    mdf["wsp_num"] = pd.to_numeric(mdf.get("wsp", np.nan), errors="coerce")
                    grp = mdf.groupby("target_date", as_index=False).agg(
                        mos_tmin_3h=("tmp_num", "min"),
                        mos_tmax_3h=("tmp_num", "max"),
                        mos_nx_min=("nx_num", "min"),
                        mos_nx_max=("nx_num", "max"),
                        mos_dpt_mean=("dpt_num", "mean"),
                        mos_wsp_mean=("wsp_num", "mean"),
                    )
                    grp["mos_tmin_f"] = grp["mos_nx_min"].combine_first(grp["mos_tmin_3h"])
                    grp["mos_tmax_f"] = grp["mos_nx_max"].combine_first(grp["mos_tmax_3h"])
                    grp["mos_id"] = m_id
                    grp["issue_date"] = today_utc
                    mos_rows.append(grp[["mos_id", "issue_date", "target_date", "mos_tmin_f", "mos_tmax_f", "mos_dpt_mean", "mos_wsp_mean"]])
        except Exception:
            continue

    live_mos_df = (
        pd.concat(mos_rows, ignore_index=True)
        if mos_rows
        else pd.DataFrame(columns=["mos_id", "issue_date", "target_date", "mos_tmin_f", "mos_tmax_f", "mos_dpt_mean", "mos_wsp_mean"])
    )
    return live_hourly, live_mos_df