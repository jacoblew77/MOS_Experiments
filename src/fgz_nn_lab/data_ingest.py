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
IEM_COOP_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/coop.py"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

ERA5_HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "snow_depth",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "boundary_layer_height",
]

NWP_ALL_CANONICAL_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "snow_depth",
    "cloud_cover",
    "cloud_cover_low",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "boundary_layer_height",
    "temperature_850hPa",
    "temperature_700hPa",
    "temperature_500hPa",
]

PREV_RUNS_BASE_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
]


def _build_prev_runs_var_list(max_prev_day: int) -> list[str]:
    out: list[str] = []
    for base_v in PREV_RUNS_BASE_VARS:
        out.append(base_v)  # Day 1 (0..24h lead)
        for k in range(1, max_prev_day + 1):
            out.append(f"{base_v}_previous_day{k}")
    return out


NWP_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "hrrr": {
        "model_code": "ncep_hrrr_conus",
        "vars": NWP_ALL_CANONICAL_VARS,
        "prev_max_day": 1,  # HRRR covers Days 1-2 (day0 + previous_day1)
    },
    "gfs": {
        "model_code": "gfs_seamless",
        "vars": NWP_ALL_CANONICAL_VARS,
        "prev_max_day": 4,  # Days 1-5 (day0 + previous_day1..4)
    },
    "ifs": {
        "model_code": "ecmwf_ifs",
        "vars": [v for v in NWP_ALL_CANONICAL_VARS if v != "boundary_layer_height"],
        "prev_max_day": 4,
    },
}

MINIMAL_FALLBACK_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def load_station_catalog(config_path: Path) -> pd.DataFrame:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = payload.get("elevation_overrides_ft", {})
    rows: list[dict[str, Any]] = []
    for st in payload["stations"]:
        sid = str(st["station_id"]).strip().upper()
        if sid == "AZ0000" or sid.startswith("AZC"):
            continue
        elev_ft = float(overrides.get(sid, st["elev_ft"]))
        rows.append(
            {
                "station_id": sid,
                "name": st["name"],
                "lat": float(st["lat"]),
                "lon": float(st["lon"]),
                "elev_ft": elev_ft,
                "elev_m": elev_ft * FT_TO_M,
                "tier": st.get("tier", "basin_plateau"),
                "obs_hour_mst": int(st.get("obs_hour_mst", 7)),
            }
        )
    return pd.DataFrame(rows)


def filter_coop_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce strict COOP QC: exclude synthetic IDs and any estimated temperatures."""
    out = df.copy()
    out["station"] = out["station"].astype(str).str.strip().str.upper()
    out = out[(out["station"] != "AZ0000") & (~out["station"].str.startswith("AZC"))].copy()

    if "temp_estimated" in out.columns:
        est_norm = (
            out["temp_estimated"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["false", "f", "0", "no", "n"])
        )
        out = out[est_norm].copy()

    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out = out.dropna(subset=["high", "low"]).copy()
    out = out[(out["high"] >= -40.0) & (out["high"] <= 135.0)].copy()
    out = out[(out["low"] >= -60.0) & (out["low"] <= 115.0)].copy()
    out = out[out["low"] <= out["high"]].copy()
    return out


def fetch_iem_coop_observations(
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    min_clean_days: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (qc_df, audit_df) where audit_df records raw vs QC-surviving row counts
    for every requested station so dropped stations are completely transparent.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_file = raw_dir / f"coop_qc_v2_{start_date}_{end_date}_{min_clean_days}d.parquet"
    audit_file = raw_dir / f"coop_audit_v2_{start_date}_{end_date}_{min_clean_days}d.parquet"
    if cache_file.exists() and audit_file.exists():
        df_cached = pd.read_parquet(cache_file)
        audit_cached = pd.read_parquet(audit_file)
        print(
            f"  [{_ts()}] [COOP] Cache HIT -> {cache_file.name} "
            f"({len(df_cached):,} rows across {df_cached['station_id'].nunique()}/{len(stations_df)} stations)",
            flush=True,
        )
        return df_cached, audit_cached

    dt_start = pd.Timestamp(start_date)
    dt_end = pd.Timestamp(end_date)
    station_ids = stations_df["station_id"].tolist()

    print(
        f"  [{_ts()}] [COOP] Downloading IEM AZCLIMATE observations for {len(station_ids)} stations "
        f"({start_date} -> {end_date})...",
        flush=True,
    )
    t0 = time.monotonic()

    params = {
        "network": "AZCLIMATE",
        "stations": ",".join(station_ids),
        "year1": dt_start.year,
        "month1": dt_start.month,
        "day1": dt_start.day,
        "year2": dt_end.year,
        "month2": dt_end.month,
        "day2": dt_end.day,
        "vars": "high,low,precip,snow,snowd,temp_estimated",
        "what": "download",
        "delim": "comma",
        "with_header": "yes",
    }

    resp = requests.get(IEM_COOP_URL, params=params, timeout=90)
    resp.raise_for_status()

    lines = [
        ln for ln in resp.text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    raw_df = pd.read_csv(io.StringIO("\n".join(lines)))
    raw_df.columns = [c.strip().lower() for c in raw_df.columns]
    raw_df["station_norm"] = raw_df["station"].astype(str).str.strip().str.upper()

    qc_df = filter_coop_qc(raw_df)
    qc_df["day"] = pd.to_datetime(qc_df["day"]).dt.tz_localize(None)
    qc_df = qc_df.rename(
        columns={
            "station": "station_id",
            "day": "obs_date",
            "high": "tmax_f",
            "low": "tmin_f",
        }
    )
    for col in ["precip", "snow", "snowd"]:
        if col in qc_df.columns:
            qc_df[col] = pd.to_numeric(qc_df[col], errors="coerce").fillna(0.0)
        else:
            qc_df[col] = 0.0

    raw_counts = raw_df.groupby("station_norm")["day"].count().to_dict()
    qc_counts = qc_df.groupby("station_id")["obs_date"].count().to_dict()

    audit_rows = []
    for _, st in stations_df.iterrows():
        sid = st["station_id"]
        r_cnt = int(raw_counts.get(sid, 0))
        q_cnt = int(qc_counts.get(sid, 0))
        kept = q_cnt >= min_clean_days
        reason = "active_kept" if kept else (
            "no_iem_rows_in_window" if r_cnt == 0 else f"below_min_clean_days({q_cnt}<{min_clean_days})"
        )
        audit_rows.append(
            {
                "station_id": sid,
                "name": st["name"],
                "elev_ft": float(st["elev_ft"]),
                "tier": st["tier"],
                "raw_days": r_cnt,
                "qc_clean_days": q_cnt,
                "min_required_days": min_clean_days,
                "kept": kept,
                "status_reason": reason,
            }
        )
    audit_df = pd.DataFrame(audit_rows)

    valid_sids = audit_df[audit_df["kept"]]["station_id"].tolist()
    qc_df = qc_df[qc_df["station_id"].isin(valid_sids)].sort_values(
        ["station_id", "obs_date"]
    ).reset_index(drop=True)

    qc_df.to_parquet(cache_file, index=False)
    audit_df.to_parquet(audit_file, index=False)

    dropped = audit_df[~audit_df["kept"]]
    print(
        f"  [{_ts()}] [COOP] Saved {len(qc_df):,} QC-clean rows across {len(valid_sids)}/{len(stations_df)} stations "
        f"in {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    if not dropped.empty:
        for _, dr in dropped.iterrows():
            print(
                f"    [COOP Audit] Excluded {dr['station_id']} ({dr['name']}): {dr['status_reason']} "
                f"[raw={dr['raw_days']}, clean={dr['qc_clean_days']}]",
                flush=True,
            )
    return qc_df, audit_df


def _request_open_meteo_json(
    session: requests.Session,
    src_label: str,
    url: str,
    params: dict[str, Any],
    retries: int = 6,
    read_timeout: int = 120,
) -> Any:
    delay = 3.0
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=(15, read_timeout))
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 and attempt < retries - 1:
                retry_after = float(resp.headers.get("Retry-After", delay * 2.0))
                wait_s = min(max(retry_after, delay), 65.0)
                print(
                    f"    [{_ts()}] [{src_label}] HTTP 429 rate-limit; sleeping {wait_s:.1f}s "
                    f"(attempt {attempt + 1}/{retries})...",
                    flush=True,
                )
                time.sleep(wait_s)
                delay = min(delay * 2.0, 60.0)
                continue
            if resp.status_code in (500, 502, 503, 504) and attempt < retries - 1:
                print(
                    f"    [{_ts()}] [{src_label}] HTTP {resp.status_code}; retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{retries})...",
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 45.0)
                continue
            resp.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt < retries - 1:
                print(
                    f"    [{_ts()}] [{src_label}] {type(exc).__name__}; retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{retries})...",
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 45.0)
                continue
            raise
    raise RuntimeError(f"Open-Meteo request failed for {url}")


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
    sleep_sec: float = 0.8,
    fallback_vars: list[str] | None = None,
) -> pd.DataFrame:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    windows = _build_date_windows(start_date, end_date, chunk_months)
    total_chunks = len(windows)

    sids = stations_df["station_id"].tolist()
    lats_csv = ",".join(f"{float(x):.4f}" for x in stations_df["lat"])
    lons_csv = ",".join(f"{float(x):.4f}" for x in stations_df["lon"])
    elev_csv = ",".join(["nan"] * len(sids))

    chunk_frames: list[pd.DataFrame] = []
    src_t0 = time.monotonic()

    for idx, (s_str, e_str) in enumerate(windows, start=1):
        chunk_file = chunk_dir / f"chunk_v2_{src_label}_{s_str}_{e_str}_{len(sids)}st.parquet"

        if chunk_file.exists():
            print(
                f"  [{_ts()}] [{src_label.upper():9s} {idx:02d}/{total_chunks:02d}] "
                f"Cache HIT  {s_str} -> {e_str}",
                flush=True,
            )
            chunk_frames.append(pd.read_parquet(chunk_file))
            continue

        print(
            f"  [{_ts()}] [{src_label.upper():9s} {idx:02d}/{total_chunks:02d}] "
            f"Requesting {s_str} -> {e_str} ({len(sids)} stations batched, {len(requested_vars)} vars)...",
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

        fb = fallback_vars or MINIMAL_FALLBACK_VARS
        try:
            payload = _request_open_meteo_json(session, src_label, url, params)
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 400 and active_vars != fb:
                print(
                    f"    [{_ts()}] [{src_label.upper()}] HTTP 400 on full var list for {s_str}..{e_str}; "
                    f"retrying fallback vars...",
                    flush=True,
                )
                active_vars = list(fb)
                params["hourly"] = ",".join(active_vars)
                payload = _request_open_meteo_json(session, src_label, url, params)
            else:
                raise

        items = payload if isinstance(payload, list) else [payload]
        if len(items) != len(sids):
            raise RuntimeError(f"Expected {len(sids)} station payloads from Open-Meteo, got {len(items)}")

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
        dt_chunk = time.monotonic() - t0
        total_elapsed = time.monotonic() - src_t0
        print(
            f"    [{_ts()}] [{src_label.upper():9s} {idx:02d}/{total_chunks:02d}] "
            f"Saved {len(chunk_df):,} rows in {dt_chunk:.1f}s (source elapsed: {total_elapsed:.0f}s)",
            flush=True,
        )
        time.sleep(sleep_sec)

    full_df = (
        pd.concat(chunk_frames, ignore_index=True)
        .drop_duplicates(subset=["station_id", "time_utc"])
        .sort_values(["station_id", "time_utc"])
        .reset_index(drop=True)
    )
    return full_df


def fetch_hourly_predictors_for_stations(
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    source_mode: str = "era5",
) -> dict[str, pd.DataFrame]:
    """
    source_mode options:
      - 'era5': 10-year ERA5 archive
      - 'nwp_hybrid': ERA5 + Historical Forecast API (HRRR, GFS, IFS)
      - 'true_lead_nwp': Historical Forecast Day-0/upper-air + Previous Runs API (_previous_day1..4)
                         for genuine 24h..120h lead-time decay evaluation.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = raw_dir / "chunks"
    pad_start = (pd.Timestamp(start_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    pad_end = (pd.Timestamp(end_date) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    if source_mode == "era5":
        sources_to_pull = ["era5"]
    elif source_mode == "nwp_hybrid":
        sources_to_pull = ["era5", "hrrr", "gfs", "ifs"]
    elif source_mode == "true_lead_nwp":
        sources_to_pull = ["era5", "hrrr", "gfs", "ifs", "prev_hrrr", "prev_gfs", "prev_ifs"]
    else:
        raise ValueError(f"Unsupported source_mode: {source_mode}")

    st_hash = "_".join(sorted(stations_df["station_id"].tolist()))
    collected: dict[str, pd.DataFrame] = {}

    with requests.Session() as session:
        session.headers.update({"Accept-Encoding": "gzip, deflate", "User-Agent": "fgz-nn-lab/0.4"})
        for s_i, src in enumerate(sources_to_pull, start=1):
            src_cache = raw_dir / f"hourly_v2_{src}_{pad_start}_{pad_end}_{len(stations_df)}st_{abs(hash(st_hash)) % 100000}.parquet"
            print(
                f"  [{_ts()}] === Source [{s_i}/{len(sources_to_pull)}]: {src.upper()} "
                f"({pad_start} -> {pad_end}) ===",
                flush=True,
            )
            if src_cache.exists():
                df_cached = pd.read_parquet(src_cache)
                print(
                    f"  [{_ts()}] [{src.upper()}] Full cache HIT -> {src_cache.name} ({len(df_cached):,} rows)",
                    flush=True,
                )
                collected[src] = df_cached
                continue

            if src == "era5":
                full_src_df = _fetch_batched_multi_station_chunks(
                    session=session,
                    src_label="era5",
                    url=OPEN_METEO_ARCHIVE_URL,
                    stations_df=stations_df,
                    start_date=pad_start,
                    end_date=pad_end,
                    requested_vars=ERA5_HOURLY_VARS,
                    canonical_vars=ERA5_HOURLY_VARS,
                    chunk_dir=chunk_dir,
                    chunk_months=12,
                    sleep_sec=0.6,
                )
            elif src.startswith("prev_"):
                base_mod = src.replace("prev_", "")
                spec = NWP_MODEL_SPECS[base_mod]
                prev_vars = _build_prev_runs_var_list(int(spec["prev_max_day"]))
                fb_prev = [
                    f"temperature_2m_previous_day{k}" if k > 0 else "temperature_2m"
                    for k in range(int(spec["prev_max_day"]) + 1)
                ] + [
                    f"dew_point_2m_previous_day{k}" if k > 0 else "dew_point_2m"
                    for k in range(int(spec["prev_max_day"]) + 1)
                ]
                full_src_df = _fetch_batched_multi_station_chunks(
                    session=session,
                    src_label=src,
                    url=OPEN_METEO_PREV_RUNS_URL,
                    stations_df=stations_df,
                    start_date=pad_start,
                    end_date=pad_end,
                    requested_vars=prev_vars,
                    canonical_vars=prev_vars,
                    chunk_dir=chunk_dir,
                    extra_params={"models": spec["model_code"]},
                    chunk_months=4,
                    sleep_sec=0.9,
                    fallback_vars=fb_prev,
                )
            else:
                spec = NWP_MODEL_SPECS[src]
                full_src_df = _fetch_batched_multi_station_chunks(
                    session=session,
                    src_label=src,
                    url=OPEN_METEO_HIST_FORECAST_URL,
                    stations_df=stations_df,
                    start_date=pad_start,
                    end_date=pad_end,
                    requested_vars=spec["vars"],
                    canonical_vars=NWP_ALL_CANONICAL_VARS,
                    chunk_dir=chunk_dir,
                    extra_params={"models": spec["model_code"]},
                    chunk_months=6,
                    sleep_sec=0.9,
                )

            full_src_df.to_parquet(src_cache, index=False)
            print(
                f"  [{_ts()}] [{src.upper()}] Completed & wrote {src_cache.name} ({len(full_src_df):,} rows)",
                flush=True,
            )
            collected[src] = full_src_df

    return collected