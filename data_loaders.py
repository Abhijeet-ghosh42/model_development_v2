# data_loaders.py

import os
import time
import datetime as dt
from typing import List, Dict, Any, Optional

import pandas as pd

try:
    import requests
except Exception as e:
    raise ImportError("The 'requests' package is required. Install with: pip install requests") from e


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
