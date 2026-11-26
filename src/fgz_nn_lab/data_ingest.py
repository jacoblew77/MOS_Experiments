from __future__ import annotations

import io
import json
import time
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

NWP_HOURLY_VARS_BASE = [
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

NWP_MODELS = {
    "hrrr": "ncep_hrrr_conus",
    "gfs": "gfs_seamless",
    "ifs": "ecmwf_ifs025",
}


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
        return pd.read_parquet(cache_file)

    dt_start = pd.Timestamp(start_date)
    dt_end = pd.Timestamp(end_date)
    station_ids = stations_df["station_id"].tolist()

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
    return qc_df


def _request_open_meteo_json(url: str, params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=(15, 150))
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            resp.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            raise
    raise RuntimeError(f"Open-Meteo request failed for {url}")


def _fetch_chunked_hourly(
    url: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    hourly_vars: list[str],
    extra_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Fetch hourly series in 1-year chunks with elevation=nan to preserve true grid elevation."""
    dt_start = pd.Timestamp(start_date)
    dt_end = pd.Timestamp(end_date)
    frames: list[pd.DataFrame] = []
    grid_elev_m = np.nan

    cur = dt_start
    while cur <= dt_end:
        chunk_end = min(cur + pd.DateOffset(years=1) - pd.Timedelta(days=1), dt_end)
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "start_date": cur.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "hourly": ",".join(hourly_vars),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
            "elevation": "nan",
        }
        if extra_params:
            params.update(extra_params)

        payload = _request_open_meteo_json(url, params)
        if np.isnan(grid_elev_m) and "elevation" in payload and payload["elevation"] is not None:
            grid_elev_m = float(payload["elevation"])

        hourly = payload.get("hourly", {})
        cdf = pd.DataFrame({"time_utc": pd.to_datetime(hourly["time"])})
        for v in hourly_vars:
            if v in hourly:
                cdf[v] = pd.to_numeric(hourly[v], errors="coerce")
            else:
                cdf[v] = np.nan
        frames.append(cdf)
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.6)

    full_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time_utc"]).sort_values("time_utc")
    full_df["grid_elev_m"] = grid_elev_m
    return full_df.reset_index(drop=True), grid_elev_m


def fetch_hourly_predictors_for_stations(
    stations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    source_mode: str = "era5",
) -> dict[str, pd.DataFrame]:
    """
    Returns dict mapping source_name -> DataFrame of hourly predictors across all stations.
    source_mode='era5' fetches ERA5 archive.
    source_mode='nwp_hybrid' fetches ERA5 + archived HRRR, GFS, and ECMWF IFS runs.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    pad_end = (pd.Timestamp(end_date) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    sources_to_pull = ["era5"]
    if source_mode == "nwp_hybrid":
        sources_to_pull.extend(["hrrr", "gfs", "ifs"])

    collected: dict[str, pd.DataFrame] = {}

    for src in sources_to_pull:
        src_cache = raw_dir / f"hourly_{src}_{start_date}_{pad_end}_{len(stations_df)}st.parquet"
        if src_cache.exists():
            collected[src] = pd.read_parquet(src_cache)
            continue

        st_frames: list[pd.DataFrame] = []
        for _, st in stations_df.iterrows():
            sid = st["station_id"]
            if src == "era5":
                df_st, _ = _fetch_chunked_hourly(
                    url=OPEN_METEO_ARCHIVE_URL,
                    lat=float(st["lat"]),
                    lon=float(st["lon"]),
                    start_date=start_date,
                    end_date=pad_end,
                    hourly_vars=ERA5_HOURLY_VARS,
                )
            else:
                model_code = NWP_MODELS[src]
                try:
                    df_st, _ = _fetch_chunked_hourly(
                        url=OPEN_METEO_HIST_FORECAST_URL,
                        lat=float(st["lat"]),
                        lon=float(st["lon"]),
                        start_date=start_date,
                        end_date=pad_end,
                        hourly_vars=NWP_HOURLY_VARS_BASE,
                        extra_params={"models": model_code},
                    )
                except requests.HTTPError:
                    minimal_vars = [
                        "temperature_2m",
                        "dew_point_2m",
                        "precipitation",
                        "cloud_cover",
                        "shortwave_radiation",
                        "wind_speed_10m",
                    ]
                    df_st, _ = _fetch_chunked_hourly(
                        url=OPEN_METEO_HIST_FORECAST_URL,
                        lat=float(st["lat"]),
                        lon=float(st["lon"]),
                        start_date=start_date,
                        end_date=pad_end,
                        hourly_vars=minimal_vars,
                        extra_params={"models": model_code},
                    )
                    for missing_col in NWP_HOURLY_VARS_BASE:
                        if missing_col not in df_st.columns:
                            df_st[missing_col] = np.nan

            df_st["station_id"] = sid
            st_frames.append(df_st)

        full_src_df = pd.concat(st_frames, ignore_index=True)
        full_src_df.to_parquet(src_cache, index=False)
        collected[src] = full_src_df

    return collected
