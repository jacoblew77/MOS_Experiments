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

NWP_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "hrrr": {
        "model_code": "ncep_hrrr_conus",
        "vars": [
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
        ],
    },
    "gfs": {
        "model_code": "gfs_seamless",
        "vars": [
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
        ],
    },
    "ifs": {
        "model_code": "ecmwf_ifs",
        "vars": [
            "temperature_2m",
            "dew_point_2m",
            "precipitation",
            "snow_depth",
            "cloud_cover",
            "cloud_cover_low",
            "shortwave_radiation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "temperature_850hPa",
            "temperature_700hPa",
            "temperature_500hPa",
        ],
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
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_file = raw_dir / f"coop_qc_{start_date}_{end_date}.parquet"
    if cache_file.exists():
        df_cached = pd.read_parquet(cache_file)
        print(
            f"  [{_ts()}] [COOP] Cache HIT -> {cache_file.name} "
            f"({len(df_cached):,} rows across {df_cached['station_id'].nunique()} stations)",
            flush=True,
        )
        return df_cached

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

    counts = qc_df.groupby("station_id")["obs_date"].count()
    valid_sids = counts[counts >= min_clean_days].index.tolist()
    qc_df = qc_df[qc_df["station_id"].isin(valid_sids)].sort_values(
        ["station_id", "obs_date"]
    ).reset_index(drop=True)

    qc_df.to_parquet(cache_file, index=False)
    print(
        f"  [{_ts()}] [COOP] Saved {len(qc_df):,} QC-clean rows across {len(valid_sids)} active stations "
        f"in {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return qc_df


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
) -> pd.DataFrame:
    """
    Fetches ALL stations simultaneously in a single multi-coordinate Open-Meteo call per time chunk,
    streaming clear progress updates on every chunk and caching each chunk immediately.
    """
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
        chunk_file = chunk_dir / f"chunk_{src_label}_{s_str}_{e_str}_{len(sids)}st.parquet"

        if chunk_file.exists():
            print(
                f"  [{_ts()}] [{src_label.upper():4s} {idx:02d}/{total_chunks:02d}] "
                f"Cache HIT  {s_str} -> {e_str}",
                flush=True,
            )
            chunk_frames.append(pd.read_parquet(chunk_file))
            continue

        print(
            f"  [{_ts()}] [{src_label.upper():4s} {idx:02d}/{total_chunks:02d}] "
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

        try:
            payload = _request_open_meteo_json(session, src_label, url, params)
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 400 and active_vars != MINIMAL_FALLBACK_VARS:
                print(
                    f"    [{_ts()}] [{src_label.upper()}] HTTP 400 on full var list for {s_str}..{e_str}; "
                    f"retrying core vars...",
                    flush=True,
                )
                active_vars = list(MINIMAL_FALLBACK_VARS)
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
            f"    [{_ts()}] [{src_label.upper():4s} {idx:02d}/{total_chunks:02d}] "
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
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = raw_dir / "chunks"
    pad_end = (pd.Timestamp(end_date) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    sources_to_pull = ["era5"]
    if source_mode == "nwp_hybrid":
        sources_to_pull.extend(["hrrr", "gfs", "ifs"])

    collected: dict[str, pd.DataFrame] = {}

    with requests.Session() as session:
        session.headers.update({"Accept-Encoding": "gzip, deflate", "User-Agent": "fgz-nn-lab/0.3"})
        for s_i, src in enumerate(sources_to_pull, start=1):
            src_cache = raw_dir / f"hourly_{src}_{start_date}_{pad_end}_{len(stations_df)}st.parquet"
            print(
                f"  [{_ts()}] === Source [{s_i}/{len(sources_to_pull)}]: {src.upper()} "
                f"({start_date} -> {pad_end}) ===",
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
                    start_date=start_date,
                    end_date=pad_end,
                    requested_vars=ERA5_HOURLY_VARS,
                    canonical_vars=ERA5_HOURLY_VARS,
                    chunk_dir=chunk_dir,
                    chunk_months=12,
                    sleep_sec=0.6,
                )
            else:
                spec = NWP_MODEL_SPECS[src]
                full_src_df = _fetch_batched_multi_station_chunks(
                    session=session,
                    src_label=src,
                    url=OPEN_METEO_HIST_FORECAST_URL,
                    stations_df=stations_df,
                    start_date=start_date,
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
