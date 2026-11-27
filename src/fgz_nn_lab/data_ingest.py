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

PREV_GFS_BASE_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
]

PREV_COMPACT_BASE_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "cloud_cover",
    "wind_speed_10m",
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


class OpenMeteoDailyQuotaError(RuntimeError):
    """Raised when a runner IP hits Open-Meteo's 24h UTC limit (Retry-After > 1200s)."""


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def _det_sig(items: list[str]) -> int:
    return sum((i + 1) * sum(ord(c) for c in s) for i, s in enumerate(items)) % 100000


def _classify_fgz_tier(sid: str, name: str, elev_ft: float) -> str:
    up = f"{sid} {name}".upper()
    if any(k in up for k in ("PHANTOM", "PHRA3", "PAGE", "PGA", "CANYON", "COTTONWOOD", "P52", "CHELLY", "CHNA3")):
        if elev_ft < 5600.0:
            return "canyon"
    if any(k in up for k in ("RIM", "PAYSON", "PAN", "PRESCOTT", "PRC", "SEDONA", "SEZ", "SHOW LOW", "SOW", "GRAND CANYON", "GCN", "ALPINE", "ALPA3")):
        return "rim_slope"
    return "basin_plateau"


def _build_prev_runs_var_list(max_prev_day: int, compact: bool = False) -> list[str]:
    base_list = PREV_COMPACT_BASE_VARS if compact else PREV_GFS_BASE_VARS
    out: list[str] = []
    for bv in base_list:
        out.append(bv)
        for k in range(1, max_prev_day + 1):
            out.append(f"{bv}_previous_day{k}")
    return out


def _nearest_asos_mos_id(lat: float, lon: float, asos_rows: list[dict[str, Any]]) -> str:
    core_mos = [r for r in asos_rows if str(r["mos_id"]) in {"KFLG", "KPRC", "KINW", "KSJN", "KGCN", "KPGA", "KRQE", "KSOW"}]
    pool = core_mos if core_mos else asos_rows
    if not pool:
        return "KFLG"
    best_id = "KFLG"
    best_d2 = 1e12
    for r in pool:
        d2 = (lat - float(r["lat"])) ** 2 + (lon - float(r["lon"])) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_id = str(r["mos_id"])
    return best_id


def load_station_catalog(config_path: Path, raw_dir: Path | None = None) -> pd.DataFrame:
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
                        for f in r.json().get("features", []):
                            props = f.get("properties", {})
                            coords = f.get("geometry", {}).get("coordinates", [None, None])
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
            mos_id = _nearest_asos_mos_id(float(d["lat"]), float(d["lon"]), asos_seeds)
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
    max_coop: int = 14,
    max_asos: int = 12,
    check_estimated_txt: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = f"{start_date}_{end_date}_{min_clean_days}d_{max_asos}a_{max_coop}c_{len(stations_df)}cand"
    obs_cache = raw_dir / f"iem_real_daily_v4_{cache_tag}.parquet"
    aud_cache = raw_dir / f"iem_real_audit_v4_{cache_tag}.parquet"
    sel_cache = raw_dir / f"iem_selected_stations_v4_{cache_tag}.parquet"

    if obs_cache.exists() and aud_cache.exists() and sel_cache.exists():
        obs_df = pd.read_parquet(obs_cache)
        aud_df = pd.read_parquet(aud_cache)
        sel_df = pd.read_parquet(sel_cache)
        print(
            f"  [{_ts()}] [OBS] Cache HIT -> {len(obs_df):,} un-estimated rows across "
            f"{len(sel_df)} FGZ sites ({(sel_df['network']=='AZ_ASOS').sum()} ASOS, {(sel_df['network']=='AZ_COOP').sum()} COOP)",
            flush=True,
        )
        return obs_df, aud_df, sel_df

    dt_start = pd.Timestamp(start_date)
    dt_end = pd.Timestamp(end_date)
    recent_cutoff = dt_end - pd.Timedelta(days=365)

    net_frames: list[pd.DataFrame] = []
    raw_count_by_uid: dict[str, int] = {}
    iem_col = "iem_sid" if "iem_sid" in stations_df.columns else "station_id"

    st_query_df = stations_df.copy()
    if "uid" not in st_query_df.columns:
        st_query_df["uid"] = st_query_df["station_id"]
    st_query_df["_query_sid"] = np.where(
        (st_query_df["network"] == "AZ_ASOS") & st_query_df[iem_col].str.startswith("K") & (st_query_df[iem_col].str.len() == 4),
        st_query_df[iem_col].str[1:],
        st_query_df[iem_col],
    )

    for net_i, (net_name, net_grp) in enumerate(st_query_df.groupby("network")):
        if net_i > 0:
            time.sleep(2.2)
        sids = net_grp["_query_sid"].tolist()
        uid_map = dict(zip(net_grp["_query_sid"], net_grp["uid"]))
        print(
            f"  [{_ts()}] [OBS] Querying IEM daily.py (na=blank, 0% estimated) for {net_name} "
            f"({len(sids)} stations, {start_date}..{end_date})...",
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

    coop_est_bad: set[tuple[str, str]] = set()
    detected_obs_hr: dict[str, int] = {}
    if check_estimated_txt and not all_qc.empty and (st_query_df["network"] == "AZ_COOP").any():
        for yr in range(max(dt_start.year, dt_end.year - 2), dt_end.year + 1):
            for st_r in st_query_df[st_query_df["network"] == "AZ_COOP"].itertuples():
                try:
                    r_txt = requests.get(
                        IEM_DAILY_API_TXT_URL,
                        params={"network": "AZ_COOP", "station": st_r._query_sid, "year": yr},
                        timeout=20,
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
        drop_est = [(u, d) in coop_est_bad for u, d in zip(all_qc["uid"].astype(str), date_strs)]
        if any(drop_est):
            print(f"  [{_ts()}] [OBS QC] Purged {int(sum(drop_est))} rows flagged temp_estimated=True", flush=True)
            all_qc = all_qc[~np.array(drop_est)].copy()

    qc_counts = all_qc.groupby("uid")["obs_date"].count().to_dict() if not all_qc.empty else {}
    rec_counts = (
        all_qc[all_qc["obs_date"] >= recent_cutoff].groupby("uid")["obs_date"].count().to_dict()
        if not all_qc.empty
        else {}
    )

    st_work = st_query_df.copy()
    for uid_k, hr_v in detected_obs_hr.items():
        st_work.loc[st_work["uid"] == uid_k, "obs_hour_mst"] = hr_v

    st_work["raw_days"] = st_work["uid"].map(lambda u: int(raw_count_by_uid.get(u, 0)))
    st_work["qc_clean_days"] = st_work["uid"].map(lambda u: int(qc_counts.get(u, 0)))
    st_work["recent_365d_clean"] = st_work["uid"].map(lambda u: int(rec_counts.get(u, 0)))
    st_work["eligible"] = (st_work["qc_clean_days"] >= min_clean_days) & (
        st_work["recent_365d_clean"] >= min_recent_days
    )

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
                "iem_sid": r["_query_sid"],
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

    selected_df = st_work[st_work["uid"].isin(kept_uids)].drop(columns=["_query_sid"]).copy()
    selected_df["iem_sid"] = st_work.loc[selected_df.index, "_query_sid"]
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


def _parse_mos_csv_text(csv_text: str, m_id: str) -> pd.DataFrame:
    lines = [ln for ln in csv_text.splitlines() if ln.strip() and not ln.startswith("#")]
    if len(lines) <= 1:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines)), low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if "runtime" not in df.columns or "ftime" not in df.columns:
        return pd.DataFrame()

    df["runtime_utc"] = pd.to_datetime(df["runtime"], utc=True, errors="coerce").dt.tz_localize(None)
    df["ftime_utc"] = pd.to_datetime(df["ftime"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["runtime_utc", "ftime_utc"]).copy()
    df = df[df["runtime_utc"].dt.hour == 12].copy()
    if df.empty:
        return pd.DataFrame()

    df["issue_date"] = df["runtime_utc"].dt.floor("D")
    df["target_date"] = (df["ftime_utc"] - pd.Timedelta(hours=7)).dt.floor("D")
    df["tmp_num"] = pd.to_numeric(df.get("tmp", np.nan), errors="coerce")
    df["nx_num"] = pd.to_numeric(df.get("n_x", np.nan), errors="coerce")
    # MEX uses 'x_n' or 'n_x' depending on bulletin format
    if "x_n" in df.columns:
        df["nx_num"] = df["nx_num"].combine_first(pd.to_numeric(df["x_n"], errors="coerce"))
    df["dpt_num"] = pd.to_numeric(df.get("dpt", np.nan), errors="coerce")
    df["wsp_num"] = pd.to_numeric(df.get("wsp", np.nan), errors="coerce")

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
    return grp[["mos_id", "issue_date", "target_date", "mos_tmin_f", "mos_tmax_f", "mos_dpt_mean", "mos_wsp_mean"]]


def fetch_iem_mos_bulletins(
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Fetches BOTH 12Z Short-Range GFS MOS ('GFS', Days 1-3) AND 12Z GFS Extended MOS ('MEX', Days 4-5)
    from IEM so mos_bulletin_* channels have full 5-day (24h..120h) coverage!
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    mos_ids = sorted(stations_df["mos_id"].dropna().unique().tolist())
    mos_cache = raw_dir / f"iem_gfs_mex_mos_v4_{start_date}_{end_date}_{_det_sig(mos_ids)}.parquet"
    if mos_cache.exists():
        mdf = pd.read_parquet(mos_cache)
        print(f"  [{_ts()}] [MOS Bulletins] Cache HIT -> {mos_cache.name} ({len(mdf):,} rows)", flush=True)
        return mdf

    dt_s = pd.Timestamp(start_date)
    dt_e = pd.Timestamp(end_date) + pd.Timedelta(days=6)
    frames: list[pd.DataFrame] = []

    print(
        f"  [{_ts()}] [MOS Bulletins] Fetching 12Z GFS (MAV D1-3) + MEX (Extended D4-5) MOS for {mos_ids}...",
        flush=True,
    )
    for m_id in mos_ids:
        st_parts: list[pd.DataFrame] = []
        for mod_name in ("GFS", "MEX"):
            try:
                params = {
                    "station": m_id,
                    "model": mod_name,
                    "sts": dt_s.strftime("%Y-%m-%dT00:00Z"),
                    "ets": dt_e.strftime("%Y-%m-%dT23:59Z"),
                    "format": "csv",
                }
                r = requests.get(IEM_MOS_CGI_URL, params=params, timeout=75)
                if r.status_code == 200 and "," in r.text:
                    parsed = _parse_mos_csv_text(r.text, m_id)
                    if not parsed.empty:
                        parsed["_priority"] = 0 if mod_name == "GFS" else 1
                        st_parts.append(parsed)
                time.sleep(1.6)
            except Exception:
                time.sleep(1.2)

        if st_parts:
            # Prefer high-res 3h GFS MAV on Days 1-3, fill Days 4-5 seamlessly with 12Z MEX!
            merged_m = (
                pd.concat(st_parts, ignore_index=True)
                .sort_values(["issue_date", "target_date", "_priority"])
                .drop_duplicates(subset=["mos_id", "issue_date", "target_date"], keep="first")
                .drop(columns=["_priority"])
            )
            frames.append(merged_m)
            print(f"    [{_ts()}] [MOS] {m_id}: {len(merged_m):,} daily D1-5 projections (GFS+MEX)", flush=True)

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
    retries: int = 12,
    read_timeout: int = 150,
) -> Any:
    delay = 10.0
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=(20, read_timeout))
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                hdr_wait = float(resp.headers.get("Retry-After", 0.0) or 0.0)
                if hdr_wait > 1200.0:
                    raise OpenMeteoDailyQuotaError(
                        f"Runner IP hit Open-Meteo 24h UTC cap (Retry-After={hdr_wait:.0f}s)"
                    )
                if attempt < retries - 1:
                    wait_s = max(min(hdr_wait, 900.0), min(delay, 120.0))
                    print(
                        f"    [{_ts()}] [{src_label}] Open-Meteo 429 pause; waiting {wait_s:.0f}s "
                        f"(attempt {attempt + 1}/{retries})...",
                        flush=True,
                    )
                    time.sleep(wait_s)
                    delay = min(delay * 1.5, 120.0)
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


def _salvage_cached_stations(
    raw_dir: Path,
    chunk_dir: Path,
    src: str,
    eff_start: str,
    pad_end: str,
    target_sids: list[str],
    canonical_vars: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Reuses 100% of stations already downloaded in Iteration 3 (`hourly_v3_*` / `chunk_v3_*`)
    or earlier Iteration 4 runs so Open-Meteo is ONLY queried for newly added delta stations!
    """
    want = set(target_sids)
    covered: set[str] = set()
    frames: list[pd.DataFrame] = []

    candidates = sorted(
        list(raw_dir.glob(f"hourly_v4_{src}_{eff_start}_{pad_end}_*.parquet"))
        + list(raw_dir.glob(f"hourly_v3_{src}_{eff_start}_{pad_end}_*.parquet")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for c_path in candidates:
        try:
            df = pd.read_parquet(c_path)
            if "station_id" not in df.columns:
                continue
            avail = set(df["station_id"].unique()).intersection(want - covered)
            if avail:
                sub = df[df["station_id"].isin(avail)].copy()
                for v in canonical_vars:
                    if v not in sub.columns:
                        sub[v] = np.float32(np.nan)
                frames.append(sub)
                covered.update(avail)
            if covered == want:
                break
        except Exception:
            continue

    # Also check partial chunk_v3_* caches (e.g., prev_ifs / prev_nbm) if full file wasn't written
    if not covered and chunk_dir.exists():
        ch_files = sorted(chunk_dir.glob(f"chunk_v3_{src}_*.parquet"))
        if ch_files:
            ch_parts = []
            for cp in ch_files:
                try:
                    cdf = pd.read_parquet(cp)
                    ch_parts.append(cdf[cdf["station_id"].isin(want)])
                except Exception:
                    continue
            if ch_parts:
                comb_ch = pd.concat(ch_parts, ignore_index=True)
                # Only treat a station as fully covered if it spans most of eff_start..pad_end
                expected_hours = max(24, int((pd.Timestamp(pad_end) - pd.Timestamp(eff_start)).days * 20))
                cnts = comb_ch.groupby("station_id")["time_utc"].count()
                good_sids = set(cnts[cnts >= expected_hours].index).intersection(want)
                if good_sids:
                    sub = comb_ch[comb_ch["station_id"].isin(good_sids)].copy()
                    for v in canonical_vars:
                        if v not in sub.columns:
                            sub[v] = np.float32(np.nan)
                    frames.append(sub)
                    covered.update(good_sids)

    salvaged_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    missing_sids = [s for s in target_sids if s not in covered]
    return salvaged_df, missing_sids


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
    allow_partial_on_daily_cap: bool = False,
) -> tuple[pd.DataFrame, bool]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    windows = _build_date_windows(start_date, end_date, chunk_months)
    total_chunks = len(windows)

    sids = stations_df["station_id"].tolist()
    st_sig = _det_sig(sids)
    lats_csv = ",".join(f"{float(x):.4f}" for x in stations_df["lat"])
    lons_csv = ",".join(f"{float(x):.4f}" for x in stations_df["lon"])
    elev_csv = ",".join(["nan"] * len(sids))

    chunk_frames: list[pd.DataFrame] = []
    src_t0 = time.monotonic()
    complete = True

    for idx, (s_str, e_str) in enumerate(windows, start=1):
        chunk_file = chunk_dir / f"chunk_v4_{src_label}_{s_str}_{e_str}_{len(sids)}st_{st_sig}.parquet"
        if chunk_file.exists():
            print(f"  [{_ts()}] [{src_label.upper():9s} {idx:02d}/{total_chunks:02d}] Delta Cache HIT {s_str} -> {e_str}", flush=True)
            cdf_hit = pd.read_parquet(chunk_file)
            for v in canonical_vars:
                if v not in cdf_hit.columns:
                    cdf_hit[v] = np.float32(np.nan)
            chunk_frames.append(cdf_hit)
            continue

        print(
            f"  [{_ts()}] [{src_label.upper():9s} {idx:02d}/{total_chunks:02d}] "
            f"Fetching DELTA ({len(sids)} new stations, {len(requested_vars)} vars): {s_str} -> {e_str}...",
            flush=True,
        )
        t0 = time.monotonic()

        params: dict[str, Any] = {
            "latitude": lats_csv,
            "longitude": lons_csv,
            "start_date": s_str,
            "end_date": e_str,
            "hourly": ",".join(requested_vars),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
            "elevation": elev_csv,
        }
        if extra_params:
            params.update(extra_params)

        try:
            payload = _request_open_meteo_json(session, src_label, url, params)
        except OpenMeteoDailyQuotaError as q_err:
            if allow_partial_on_daily_cap:
                print(f"    [{_ts()}] [{src_label.upper()}] {q_err} — masking remaining optional chunks.", flush=True)
                complete = False
                break
            raise

        items = payload if isinstance(payload, list) else [payload]
        loc_dfs: list[pd.DataFrame] = []
        for sid, item in zip(sids, items):
            hourly = item.get("hourly", {})
            grid_elev = float(item["elevation"]) if item.get("elevation") is not None else np.nan
            cdf = pd.DataFrame({"time_utc": pd.to_datetime(hourly["time"])})
            for v in canonical_vars:
                cdf[v] = pd.to_numeric(hourly[v], errors="coerce").astype(np.float32) if v in hourly else np.float32(np.nan)
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

    if not chunk_frames:
        empty_cols = {v: pd.Series(dtype="float32") for v in canonical_vars}
        return pd.DataFrame({"station_id": pd.Series(dtype="str"), "time_utc": pd.Series(dtype="datetime64[ns]"), "grid_elev_m": pd.Series(dtype="float32"), **empty_cols}), False

    return (
        pd.concat(chunk_frames, ignore_index=True)
        .drop_duplicates(subset=["station_id", "time_utc"])
        .sort_values(["station_id", "time_utc"])
        .reset_index(drop=True)
    ), complete


def fetch_hourly_predictors_for_stations(
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    source_mode: str = "true_lead_nwp",
) -> dict[str, pd.DataFrame]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir_v3 = raw_dir / "chunks_v3"
    chunk_dir_v4 = raw_dir / "chunks_v4"
    pad_start = (pd.Timestamp(start_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    pad_end = (pd.Timestamp(end_date) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    if source_mode == "era5":
        if len(stations_df) >= 12 and pd.Timestamp(pad_start) < pd.Timestamp("2021-01-01"):
            pad_start = "2021-01-01"
        sources = ["era5"]
    else:
        sources = ["era5", "gfs_hist", "prev_hrrr", "prev_gfs", "prev_ifs", "prev_nbm"]

    sids_sorted = sorted(stations_df["station_id"].tolist())
    st_sig = _det_sig(sids_sorted)
    collected: dict[str, pd.DataFrame] = {}

    src_spec_map = {
        "era5": (OPEN_METEO_ARCHIVE_URL, ERA5_HOURLY_VARS, ERA5_HOURLY_VARS, None, 12, 0.8, (source_mode == "era5")),
        "gfs_hist": (OPEN_METEO_HIST_FORECAST_URL, HIST_SUPPLEMENT_VARS, HIST_SUPPLEMENT_VARS, {"models": "gfs_seamless"}, 8, 0.9, False),
        "prev_hrrr": (OPEN_METEO_PREV_RUNS_URL, _build_prev_runs_var_list(1, False), _build_prev_runs_var_list(1, False), {"models": "ncep_hrrr_conus"}, 6, 1.0, False),
        "prev_gfs": (OPEN_METEO_PREV_RUNS_URL, _build_prev_runs_var_list(4, False), _build_prev_runs_var_list(4, False), {"models": "gfs_seamless"}, 4, 1.0, False),
        "prev_ifs": (OPEN_METEO_PREV_RUNS_URL, _build_prev_runs_var_list(4, True), _build_prev_runs_var_list(4, False), {"models": "ecmwf_ifs"}, 6, 1.0, True),
        "prev_nbm": (OPEN_METEO_PREV_RUNS_URL, _build_prev_runs_var_list(4, True), _build_prev_runs_var_list(4, False), {"models": "ncep_nbm_conus"}, 6, 1.0, True),
    }

    with requests.Session() as session:
        session.headers.update({"Accept-Encoding": "gzip, deflate", "User-Agent": "fgz-nn-lab/0.7"})
        for s_i, src in enumerate(sources, start=1):
            eff_start = max(pad_start, "2024-10-05") if src == "prev_nbm" else pad_start
            if pd.Timestamp(eff_start) > pd.Timestamp(pad_end):
                continue

            src_cache = raw_dir / f"hourly_v4_{src}_{eff_start}_{pad_end}_{len(stations_df)}st_{st_sig}.parquet"
            print(f"  [{_ts()}] === Source [{s_i}/{len(sources)}]: {src.upper()} ({eff_start} -> {pad_end}) ===", flush=True)
            if src_cache.exists():
                cdf = pd.read_parquet(src_cache)
                print(f"  [{_ts()}] [{src.upper()}] Full Cache HIT -> {src_cache.name} ({len(cdf):,} rows)", flush=True)
                collected[src] = cdf
                continue

            url, req_vars, canon_vars, extra_p, c_months, slp, allow_part = src_spec_map[src]
            salvaged_df, missing_sids = _salvage_cached_stations(
                raw_dir, chunk_dir_v3, src, eff_start, pad_end, sids_sorted, canon_vars
            )
            if salvaged_df is not None and not salvaged_df.empty:
                n_salv = salvaged_df["station_id"].nunique()
                print(
                    f"  [{_ts()}] [{src.upper()}] Reused {n_salv}/{len(sids_sorted)} stations from prior cache (0 API credits)! "
                    f"Delta stations needed: {len(missing_sids)}",
                    flush=True,
                )

            if not missing_sids:
                full_df = salvaged_df.sort_values(["station_id", "time_utc"]).reset_index(drop=True)
                full_df.to_parquet(src_cache, index=False)
                collected[src] = full_df
                continue

            delta_st_df = stations_df[stations_df["station_id"].isin(missing_sids)].reset_index(drop=True)
            delta_df, is_complete = _fetch_batched_multi_station_chunks(
                session, src, url, delta_st_df, eff_start, pad_end,
                req_vars, canon_vars, chunk_dir_v4, extra_params=extra_p,
                chunk_months=c_months, sleep_sec=slp, allow_partial_on_daily_cap=allow_part,
            )

            parts = [p for p in (salvaged_df, delta_df) if p is not None and not p.empty]
            if parts:
                comb_df = (
                    pd.concat(parts, ignore_index=True)
                    .drop_duplicates(subset=["station_id", "time_utc"])
                    .sort_values(["station_id", "time_utc"])
                    .reset_index(drop=True)
                )
                if is_complete and comb_df["station_id"].nunique() == len(sids_sorted):
                    comb_df.to_parquet(src_cache, index=False)
                collected[src] = comb_df

    return collected


def fetch_live_operational_inputs(
    stations_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Lightweight Daily Operational Fetch (~106.6 Open-Meteo credits = 1.07% of daily limit!):
      - Pulls 7-day live GFS, ECMWF IFS, 3km HRRR, and 2.5km NBM grids
      - Pulls live GFS + MEX + NBS/NBE MOS JSON bulletins across all FGZ airports (Days 1-5)
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

    mos_rows: list[pd.DataFrame] = []
    today_mst = (pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(hours=7)).floor("D").tz_localize(None)

    for m_id in sorted(stations_df["mos_id"].dropna().unique().tolist()):
        st_m_parts: list[pd.DataFrame] = []
        for prio, mod_code in enumerate(("GFS", "NBS", "MEX", "NBE")):
            try:
                r = requests.get(IEM_MOS_LIVE_URL, params={"station": m_id, "model": mod_code}, timeout=15)
                if r.status_code == 200:
                    entries = r.json().get("data", [])
                    if entries:
                        mdf = pd.DataFrame(entries)
                        mdf["ftime_utc"] = pd.to_datetime(mdf["ftime"], utc=True, errors="coerce").dt.tz_localize(None)
                        mdf["target_date"] = (mdf["ftime_utc"] - pd.Timedelta(hours=7)).dt.floor("D")
                        mdf["tmp_num"] = pd.to_numeric(mdf.get("tmp", np.nan), errors="coerce")
                        mdf["nx_num"] = pd.to_numeric(mdf.get("n_x", np.nan), errors="coerce")
                        if "x_n" in mdf.columns:
                            mdf["nx_num"] = mdf["nx_num"].combine_first(pd.to_numeric(mdf["x_n"], errors="coerce"))
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
                        grp["issue_date"] = today_mst
                        grp["_p"] = prio
                        st_m_parts.append(grp[["mos_id", "issue_date", "target_date", "mos_tmin_f", "mos_tmax_f", "mos_dpt_mean", "mos_wsp_mean", "_p"]])
            except Exception:
                continue
        if st_m_parts:
            mos_rows.append(
                pd.concat(st_m_parts, ignore_index=True)
                .sort_values(["target_date", "_p"])
                .drop_duplicates(subset=["mos_id", "target_date"], keep="first")
                .drop(columns=["_p"])
            )

    live_mos_df = (
        pd.concat(mos_rows, ignore_index=True)
        if mos_rows
        else pd.DataFrame(columns=["mos_id", "issue_date", "target_date", "mos_tmin_f", "mos_tmax_f", "mos_dpt_mean", "mos_wsp_mean"])
    )
    return live_hourly, live_mos_df