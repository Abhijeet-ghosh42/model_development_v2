import os
import time
import datetime as dt
from typing import List, Dict, Any, Optional

import pandas as pd

import requests
import pandas as pd
from datetime import datetime, timedelta

"""
data_loaders.py
---------------

This module centralizes all data loading logic for the predictive maintenance
training scripts. It provides utilities to fetch data either from **local CSVs**
or from the **IoT API** (energy, harmonics, vibration). Since the API can only
serve up to 24 hours of data at once, the loaders automatically split a long
time range into smaller chunks and then stitch the results back together into
a single DataFrame.

Functions
~~~~~~~~~
- _parse_iso_z(s) / _to_iso_z(t): Helpers to convert timestamps between
  ISO8601 '...Z' strings and Python datetimes.

- _chunk_ranges(start_utc, end_utc, max_span_hours):
  Splits a long [start, end] UTC range into consecutive <=24h chunks.

- load_esa_harmonics_api(...):
  Calls the API to fetch ESA harmonics data in chunks, concatenates the JSON
  payloads, deduplicates, sorts by timestamp, and returns a DataFrame.

- load_local_csv(csv_path):
  Convenience wrapper to read a local CSV into a DataFrame.

- load_energy_api(...):
  Fetches energy (time series) data in <=24h chunks from the API, returns a
  concatenated DataFrame.

- load_vibration_api(...):
  Fetches vibration sensor data in <=24h chunks from the API, returns a
  concatenated DataFrame.

Notes
~~~~~
- All loaders return a pandas DataFrame (possibly empty if the API returns no data).
- They apply basic normalization like timestamp parsing and deduplication.
- Each script in `scripts/` can import these functions to switch between
  local CSV training or API-based training without rewriting logic.
"""


def _parse_iso_z(s: str) -> dt.datetime:
    """Parse 'YYYY-MM-DDTHH:MM:SSZ' to aware UTC datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)


def _to_iso_z(t: dt.datetime) -> str:
    """Format aware UTC datetime to 'YYYY-MM-DDTHH:MM:SSZ'."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    else:
        t = t.astimezone(dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunk_ranges(
    start_utc: str,
    end_utc: str,
    max_span_hours: int = 24,
) -> List[Dict[str, str]]:
    """Split [start,end] into consecutive chunks of <= max_span_hours."""
    start = _parse_iso_z(start_utc)
    end = _parse_iso_z(end_utc)
    if end <= start:
        return []

    chunks = []
    cur = start
    delta = dt.timedelta(hours=max_span_hours)
    while cur < end:
        nxt = min(cur + delta, end)
        chunks.append({"start": _to_iso_z(cur), "end": _to_iso_z(nxt)})
        cur = nxt
    return chunks


def load_esa_harmonics_api(
    *,
    base_url: str,
    api_key: str,
    machine_id: str | int,
    start_utc: str,
    end_utc: str,
    endpoint: str = "/api/v1/energy/",
    max_span_hours: int = 24,
    rate_limit_s: float = 0.2,
    extra_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    - Splits into <= 24h chunks.
    - Concatenates JSON payloads (list or {'data': list}).
    - Drops duplicates, sorts by 'timestamp' if present.
    """
    # Normalize URL join (avoid double slashes)
    base_url = base_url.rstrip("/")
    endpoint = endpoint.lstrip("/")
    url = f"{base_url}/{endpoint}"

    params_static = {
        "type": "harmonics",
        "duration_type": "custom",
        "response_type": "raw",
    }
    if extra_params:
        params_static.update(extra_params)

    headers = {
        "accept": "application/json",
        "api_key": api_key,
    }

    frames: List[pd.DataFrame] = []
    chunks = _chunk_ranges(start_utc, end_utc, max_span_hours=max_span_hours)

    for i, ch in enumerate(chunks, 1):
        params = {
            **params_static,
            "machine_id": str(machine_id),
            "start_time": ch["start"],
            "end_time": ch["end"],
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            # API could return a list or a dict with 'data'
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
                records = payload["data"]
            else:
                records = []

            if records:
                df_chunk = pd.json_normalize(records)
                frames.append(df_chunk)
        except Exception as e:
            
            print(f"[WARN] Chunk {i}/{len(chunks)} failed: {e}")

        time.sleep(rate_limit_s)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates()

    # Sort by timestamp if present
    ts_col = None
    for cand in ["timestamp", "ts", "created_at"]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col:
        # if parse fails we keep original
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            df = df.sort_values(ts_col).reset_index(drop=True)
        except Exception:
            pass

    return df


def load_local_csv(csv_path: str) -> pd.DataFrame:
    """Tiny convenience wrapper (kept for symmetry)."""
    return pd.read_csv(csv_path)

def load_energy_api(base_url, api_key, machine_id, start_utc, end_utc,
                    endpoint="/api/v1/energy/", max_span_hours=24, extra_params=None):
    """
    Fetch energy data from API in chunks (since API only supports ≤24h).
    Returns: pd.DataFrame
    """
    headers = {"accept": "application/json", "api_key": api_key}
    all_frames = []

    start_dt = pd.to_datetime(start_utc)
    end_dt = pd.to_datetime(end_utc)

    while start_dt < end_dt:
        batch_end = min(start_dt + timedelta(hours=max_span_hours), end_dt)

        params = {
            "type": extra_params.get("type", "energy") if extra_params else "energy",
            "duration_type": "custom",
            "response_type": "raw",
            "start_time": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": batch_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "machine_id": machine_id,
        }

        url = f"{base_url.rstrip('/')}{endpoint}"
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                all_frames.append(df)

        start_dt = batch_end  # move to next chunk

    if not all_frames:
        return pd.DataFrame()

    df_final = pd.concat(all_frames, ignore_index=True)
    return df_final

def load_vibration_api(base_url, api_key, machine_id, start_utc, end_utc,
                       endpoint="/api/v1/vibration/", max_span_hours=24, extra_params=None):
    """
    Fetch vibration data from API in 24h chunks and return as a pandas DataFrame.
    """
    headers = {"accept": "application/json", "api_key": api_key}
    start_dt = pd.to_datetime(start_utc)
    end_dt = pd.to_datetime(end_utc)
    dfs = []

    while start_dt < end_dt:
        chunk_end = min(start_dt + pd.Timedelta(hours=max_span_hours), end_dt)

        params = {
            "machine_id": machine_id,
            "start_time": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "vibration",
            "response_type": "raw"
        }
        if extra_params:
            params.update(extra_params)

        url = base_url.rstrip("/") + endpoint
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data:
            dfs.append(pd.DataFrame(data))

        start_dt = chunk_end

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()