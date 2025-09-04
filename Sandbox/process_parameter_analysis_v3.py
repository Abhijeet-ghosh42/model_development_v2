# process_parameter_analysis_v3.py
# -----------------------------------------------------------------------------
# Process data based alerts for extruder lines.
# - Minimal machine config (YAML/JSON) is optional.
# - If a config is provided, we use it for zones, column names, and thresholds.
# - If no config is provided, we auto-detect sensible defaults.
# - Alerts run on the full rolling window (not the mean of the window).
#
# I kept the config shape small and avoided unnecessary knobs to retain simplicity.
# -----------------------------------------------------------------------------

import asyncio
import json
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

try:
    import yaml  # optional; only needed if loading YAML configs
except Exception:
    yaml = None


# ----------------------------- Config Utilities ------------------------------

def _load_config(config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a small config dict from a provided object or from a file.
    We support YAML (if pyyaml is available) and JSON files. If both
    config and config_path are None, return defaults.
    """
    if config is not None:
        return config

    if config_path:
        if config_path.endswith((".yml", ".yaml")):
            if yaml is None:
                raise ImportError("pyyaml is required to load YAML config files.")
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        elif config_path.endswith(".json"):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        else:
            raise ValueError("Unsupported config extension. Use .yaml/.yml or .json.")

    # Defaults if nothing is provided
    return {
        "naming": {
            "separator": "_"  # for composing bz1_actpv etc. (kept for future extensibility)
        },
        "zones": {
            # If empty, we auto-detect 'bz*_actpv' columns.
            "bz": [],
            "include_dz": False,
            "dz": []
        },
        "columns": {
            "ext_actrpm": "ext_actrpm",
            "ext_setrpm": "ext_setrpm",
            "fdr_actrpm": "fdr_actrpm",
            "fdr_setrpm": "fdr_setrpm",
            "ext_torq": "ext_torq",
            "fdr_torq": "fdr_torq",
            "fhoff_torq": "fhoff_torq",
            "meltpres": "meltpres",
            "melttemp": "melttemp",
            "hauloff_mpm": "fhoff_actmpm"
        },
        "thresholds": {
            "setpoint_tolerance": 5,
            "stability_tolerance": 4,
            "spike_tolerance": 6,
            "z_score_threshold": 1
        },
        "mfi": {
            "C": 1.0,
            "n": 0.4,
            "fluctuation_threshold": 2.0,
            "drop_threshold": 2.0,
            "trend_window": 8,
            "trend_slope_threshold": 0.07
        },
        "shear_heat": {
            "spike_tolerance": 15.0,
            "high_threshold": 50.0
        },
        "mfcs": {
            "corr_window": 50,
            "fluctuation_threshold": 0.01,
            "spike_threshold": 0.02,
            "trend_window": 5,
            "trend_slope_threshold": -0.05
        }
    }


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Return a series if present; else an empty float series.
    """
    if col in df.columns:
        return df[col]
    return pd.Series(dtype=float)


def _polyfit_slope(y: pd.Series) -> Optional[float]:
    """
    Slope of a simple linear fit over the index range [0..len(y)-1].
    None if insufficient points.
    """
    if len(y) < 2:
        return None
    x = np.arange(len(y))
    # polyfit returns [slope, intercept]
    return float(np.polyfit(x, y.values, 1)[0])


# ----------------------------- Main Analyzer ---------------------------------

class ProcessParameterAnalysis:
    def __init__(
        self,
        window: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize with a rolling window of process parameter data (list of dicts).
        Optionally pass a config dict or a file path to a YAML/JSON config.
        """
        # Keep original time-series data for rolling/spike/trend logic
        self.original_data = pd.DataFrame(window)

        # Remove known metadata columns if present; keep time series intact
        self.df = self.original_data.copy()
        for col in ["_id", "machine_id", "tenant_id"]:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)

        # Coerce numerics where possible (non-numeric stays as-is)
        # This helps rolling/std without crashing on string columns.
        self.df = self.df.apply(pd.to_numeric, errors="ignore")

        # Load config (dict wins over file path if both provided)
        self.cfg = _load_config(config=config, config_path=config_path)

        # Resolve simple thresholds
        thr = self.cfg.get("thresholds", {})
        self.setpoint_tolerance = float(thr.get("setpoint_tolerance", 5))
        self.stability_tolerance = float(thr.get("stability_tolerance", 4))
        self.spike_tolerance = float(thr.get("spike_tolerance", 6))
        self.z_score_threshold = float(thr.get("z_score_threshold", 1))

        # MFI parameters
        mfi_cfg = self.cfg.get("mfi", {})
        self.C = float(mfi_cfg.get("C", 1.0))
        self.n = float(mfi_cfg.get("n", 0.4))
        self.mfi_fluctuation_threshold = float(mfi_cfg.get("fluctuation_threshold", 2.0))
        self.drop_threshold = float(mfi_cfg.get("drop_threshold", 2.0))
        self.trend_window = int(mfi_cfg.get("trend_window", 8))
        self.trend_slope_threshold = float(mfi_cfg.get("trend_slope_threshold", 0.07))

        # Shear-heat
        sh_cfg = self.cfg.get("shear_heat", {})
        self.shear_spike_tol = float(sh_cfg.get("spike_tolerance", 15.0))
        self.shear_high_threshold = float(sh_cfg.get("high_threshold", 50.0))

        # MFCS config
        mfcs_cfg = self.cfg.get("mfcs", {})
        self.mfcs_corr_window = int(mfcs_cfg.get("corr_window", 50))
        self.mfcs_fluctuation_threshold = float(mfcs_cfg.get("fluctuation_threshold", 0.01))
        self.mfcs_spike_threshold = float(mfcs_cfg.get("spike_threshold", 0.02))
        self.mfcs_trend_window = int(mfcs_cfg.get("trend_window", 5))
        self.mfcs_trend_slope_threshold = float(mfcs_cfg.get("trend_slope_threshold", -0.05))

        # Column map
        cols = self.cfg.get("columns", {})
        self.col_ext_actrpm = cols.get("ext_actrpm", "ext_actrpm")
        self.col_ext_setrpm = cols.get("ext_setrpm", "ext_setrpm")
        self.col_fdr_actrpm = cols.get("fdr_actrpm", "fdr_actrpm")
        self.col_fdr_setrpm = cols.get("fdr_setrpm", "fdr_setrpm")
        self.col_ext_torq = cols.get("ext_torq", "ext_torq")
        self.col_fdr_torq = cols.get("fdr_torq", "fdr_torq")
        self.col_fhoff_torq = cols.get("fhoff_torq", "fhoff_torq")
        self.col_meltpres = cols.get("meltpres", "meltpres")
        self.col_melttemp = cols.get("melttemp", "melttemp")
        self.col_hauloff_mpm = cols.get("hauloff_mpm", "fhoff_actmpm")

        # Screw & torque column lists for iteration
        self.screw_actual_columns = [self.col_ext_actrpm, self.col_fdr_actrpm]
        self.screw_setpoint_columns = [self.col_ext_setrpm, self.col_fdr_setrpm]
        self.torque_actual_columns = [self.col_ext_torq, self.col_fdr_torq, self.col_fhoff_torq]

        # Zone columns for temperature analysis
        self.actual_columns, self.setpoint_columns = self._resolve_zone_columns()

        # Derived features (time series)
        self.df["shear_heat"] = self._calculate_shear_heat_series()
        self.df["MFI"] = self._calculate_mfi_series()

        # Material Flow Consistency Score components
        self._calculate_mfcs()  # adds MFCS columns to self.df safely

    # --------------------------- Column Resolution ---------------------------

    def _resolve_zone_columns(self) -> (List[str], List[str]):
        """
        Build temp actual/setpoint lists using config if present; otherwise auto-detect
        columns matching 'bz*_actpv' and corresponding '*_setpv'.
        """
        zones_cfg = self.cfg.get("zones", {}) or {}
        bz_list = zones_cfg.get("bz", []) or []
        include_dz = bool(zones_cfg.get("include_dz", False))
        dz_list = zones_cfg.get("dz", []) or []

        actual_cols: List[str] = []
        setpoint_cols: List[str] = []

        if bz_list:
            # Use configured zone names (e.g., ["bz1", "bz2", ...])
            for z in bz_list:
                act_col = f"{z}_actpv"
                set_col = f"{z}_setpv"
                if act_col in self.df.columns:
                    actual_cols.append(act_col)
                    if set_col in self.df.columns:
                        setpoint_cols.append(set_col)
            if include_dz and dz_list:
                for z in dz_list:
                    act_col = f"{z}_actpv"
                    set_col = f"{z}_setpv"
                    if act_col in self.df.columns:
                        actual_cols.append(act_col)
                        if set_col in self.df.columns:
                            setpoint_cols.append(set_col)
        else:
            # Auto-detect bz*_actpv columns
            for c in self.df.columns:
                if isinstance(c, str) and c.startswith("bz") and c.endswith("_actpv"):
                    actual_cols.append(c)
                    set_c = c.replace("_actpv", "_setpv")
                    if set_c in self.df.columns:
                        setpoint_cols.append(set_c)

        return actual_cols, setpoint_cols

    # ---------------------------- Derived Signals ----------------------------

    def _calculate_mfi_series(self) -> pd.Series:
        """
        Melt Flow Index (MFI) series across the rolling window.
        """
        # We require melt temp (C), extruder rpm, and melt pressure.
        T_kelvin = _safe_series(self.df, self.col_melttemp) + 273.15
        ext_rpm = _safe_series(self.df, self.col_ext_actrpm)
        meltpres = _safe_series(self.df, self.col_meltpres)

        # Safe math: where values are missing, result becomes NaN and is handled upstream.
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.C * np.exp(-meltpres) * (ext_rpm ** (1 - self.n)) * np.exp(-1.0 / T_kelvin)

    def _calculate_shear_heat_series(self) -> pd.Series:
        """
        Shear heat time series across the rolling window.
        """
        meltpres = _safe_series(self.df, self.col_meltpres)
        ext_torq = _safe_series(self.df, self.col_ext_torq)
        ext_rpm = _safe_series(self.df, self.col_ext_actrpm)
        T_kelvin = _safe_series(self.df, self.col_melttemp) + 273.15

        with np.errstate(divide="ignore", invalid="ignore"):
            return self.C * meltpres * ext_torq * ext_rpm * np.exp(-1.0 / T_kelvin)

    # ------------------------------- MFCS ------------------------------------

    def _calculate_mfcs(self) -> None:
        """
        Computes Material Flow Consistency Score (MFCS) and adds columns in self.df.
        """
        # Pull series; if missing, they come back as empty and operations handle gracefully
        fdr_act = _safe_series(self.df, self.col_fdr_actrpm)
        haul_mpm = _safe_series(self.df, self.col_hauloff_mpm)
        melt_pres = _safe_series(self.df, self.col_meltpres)
        fdr_torq = _safe_series(self.df, self.col_fdr_torq)

        # PCC
        if not fdr_act.empty and not haul_mpm.empty:
            pcc = fdr_act.rolling(window=self.mfcs_corr_window).corr(haul_mpm).fillna(0).clip(0, 1)
        else:
            pcc = pd.Series([0.0] * len(self.df), index=self.df.index)

        # Haul Off Stability
        if not haul_mpm.empty:
            hos = (1 - haul_mpm.rolling(window=self.mfcs_corr_window).std() / (haul_mpm.mean() if haul_mpm.mean() else 1)).fillna(0).clip(0, 1)
        else:
            hos = pd.Series([0.0] * len(self.df), index=self.df.index)

        # Melt Pressure Stability (robust mean abs dev / mean)
        if not melt_pres.empty:
            def _mad_over_mean(x: pd.Series) -> float:
                mu = x.mean()
                if mu == 0 or np.isnan(mu):
                    return 1.0
                return np.mean(np.abs(x - mu)) / mu

            mps = (1 - melt_pres.rolling(window=self.mfcs_corr_window).apply(_mad_over_mean, raw=False)).fillna(0).clip(0, 1)
        else:
            mps = pd.Series([0.0] * len(self.df), index=self.df.index)

        # Torque Efficiency
        if not fdr_torq.empty:
            te = (fdr_act / fdr_torq.replace(0, np.nan)).fillna(0).clip(0, 1)
        else:
            te = pd.Series([0.0] * len(self.df), index=self.df.index)

        self.df["PCC"] = pcc
        self.df["Haul_Off_Stability"] = hos
        self.df["Melt_Pressure_Stability"] = mps
        self.df["Torque_Efficiency"] = te
        self.df["MFCS"] = (pcc + hos + mps + te) / 4.0

    # ------------------------------- Alerts ----------------------------------

    async def zonewise_temperature_setpoint_deviation_alert(self):
        alert_code = "temp_setpoint_deviation"
        params = []
        status = 0

        # Use the last point in the window (most recent)
        if len(self.df) == 0:
            return None

        last = self.df.iloc[-1]

        for act_col in self.actual_columns:
            set_col = act_col.replace("_actpv", "_setpv")
            if act_col in self.df.columns and set_col in self.df.columns:
                act = last.get(act_col, np.nan)
                setp = last.get(set_col, np.nan)
                if pd.notna(act) and pd.notna(setp):
                    if abs(act - setp) > self.setpoint_tolerance:
                        status = 1
                        params.extend([act_col, set_col])

        if status == 1:
            return {"code": alert_code, "params": params}

    async def zonewise_temperature_heater_stability_alert(self):
        """
        Rolling standard deviation over windows of 5; if any window exceeds threshold,
        raise stability alert for that zone.
        """
        alert_code = "temp_stability"
        params = []
        status = 0

        for col in self.actual_columns:
            if col in self.df.columns and len(self.df[col]) >= 5:
                rolling_std = self.df[col].rolling(window=5).std()
                if (rolling_std > self.stability_tolerance).any():
                    status = 1
                    params.append(col)

        if status == 1:
            return {"code": alert_code, "params": params}

    async def zonewise_temperature_spike_and_drop_alert(self):
        """
        Detects sudden spikes/drops using absolute first-difference > spike_tolerance.
        """
        alert_code = "temp_spike"
        params = []
        status = 0

        for col in self.actual_columns:
            if col in self.df.columns and len(self.df[col]) >= 2:
                diffs = self.df[col].diff().abs().iloc[1:]
                if (diffs > self.spike_tolerance).any():
                    status = 1
                    params.append(col)

        if status == 1:
            return {"code": alert_code, "params": params}

    async def screw_rotation_setpoint_deviation_alert(self):
        alert_code = "screw_setpoint_deviation"
        params = []
        status = 0

        if len(self.df) == 0:
            return None

        last = self.df.iloc[-1]
        pairs = list(zip(self.screw_actual_columns, self.screw_setpoint_columns))
        for act_col, set_col in pairs:
            if act_col in self.df.columns and set_col in self.df.columns:
                act = last.get(act_col, np.nan)
                setp = last.get(set_col, np.nan)
                if pd.notna(act) and pd.notna(setp):
                    if abs(act - setp) > self.setpoint_tolerance:
                        status = 1
                        params.append(act_col)

        if status == 1:
            return {"code": alert_code, "params": params}

    async def screw_rotation_anomaly_detection_alert(self):
        """
        Simple z-score check on the most recent value vs window mean/std.
        """
        alert_code = "screw_anomaly"
        params = []
        status = 0

        for col in self.screw_actual_columns:
            if col in self.df.columns:
                series = self.df[col].dropna()
                if len(series) >= 3:
                    mu, sigma = series.mean(), series.std()
                    last_val = series.iloc[-1]
                    if sigma and np.isfinite(sigma):
                        z = (last_val - mu) / sigma
                        if abs(z) > self.z_score_threshold:
                            status = 1
                            params.append(col)

        if status == 1:
            return {"code": alert_code, "params": params}

    async def torque_anomaly_detection_alert(self):
        alert_code = "torq_anomaly"
        params = []
        status = 0

        for col in self.torque_actual_columns:
            if col in self.df.columns:
                series = self.df[col].dropna()
                if len(series) >= 3:
                    mu, sigma = series.mean(), series.std()
                    last_val = series.iloc[-1]
                    if sigma and np.isfinite(sigma):
                        z = (last_val - mu) / sigma
                        if abs(z) > self.z_score_threshold:
                            status = 1
                            params.append(col)

        if status == 1:
            return {"code": alert_code, "params": params}

    async def torque_spike_detection_alert(self):
        alert_code = "torq_trend"
        params = []
        status = 0

        for col in self.torque_actual_columns:
            if col in self.df.columns and len(self.df[col]) >= 2:
                diffs = self.df[col].diff().abs().iloc[1:]
                if (diffs > self.spike_tolerance).any():
                    status = 1
                    params.append(col)

        if status == 1:
            return {"code": alert_code, "params": params}

    async def mfi_high_fluctuation_alert(self):
        alert_code = "mfi_fluctuation"
        status = 0
        params = []

        if "MFI" in self.df.columns:
            mfi = self.df["MFI"].dropna()
            if len(mfi) >= 5:
                std_dev = mfi.rolling(window=5).std().iloc[-1]
                if pd.notna(std_dev) and std_dev > self.mfi_fluctuation_threshold:
                    status = 1
                    params.append(f"MFI std@5={std_dev:.2f}")

        if status == 1:
            return {"code": alert_code, "params": params}

    async def mfi_sudden_drops_alert(self):
        alert_code = "mfi_drop"
        status = 0
        params = []

        if "MFI" in self.df.columns and len(self.df["MFI"]) >= 2:
            diffs = self.df["MFI"].diff().iloc[1:]
            drops = diffs[diffs < -self.drop_threshold]
            if not drops.empty:
                status = 1
                params.append(f"Drops: {drops.values.tolist()}")

        if status == 1:
            return {"code": alert_code, "params": params}

    async def mfi_gradual_rising_trend_alert(self):
        alert_code = "mfi_trend"
        status = 0
        params = []

        if "MFI" in self.df.columns and len(self.df["MFI"]) >= self.trend_window:
            window_vals = self.df["MFI"].iloc[-self.trend_window:].dropna()
            if len(window_vals) == self.trend_window:
                slope = _polyfit_slope(window_vals)
                if slope is not None and slope > self.trend_slope_threshold:
                    status = 1
                    params.append(f"{slope:.2f}")

        if status == 1:
            return {"code": alert_code, "params": params}

    async def shear_heat_spike_alert(self):
        alert_code = "shear_heat_spike"
        status = 0
        params = []

        if "shear_heat" in self.df.columns and len(self.df["shear_heat"]) >= 2:
            diffs = self.df["shear_heat"].diff().abs().iloc[1:]
            if (diffs > self.shear_spike_tol).any():
                status = 1
                params.append("shear_heat")

        if status == 1:
            return {"code": alert_code, "params": params}

    async def shear_heat_threshold_alert(self):
        alert_code = "shear_heat_high"
        status = 0
        params = []

        if "shear_heat" in self.df.columns and len(self.df["shear_heat"]) > 0:
            val = self.df["shear_heat"].iloc[-1]
            if pd.notna(val) and val > self.shear_high_threshold:
                status = 1
                params.append(f"{val:.2f}")

        if status == 1:
            return {"code": alert_code, "params": params}

    async def mfcs_fluctuation_alert(self):
        alert_code = "mfcs_fluctuation"
        status = 0
        params = ["fdr_actrpm", "fhoff_actmpm", "meltpres", "fdr_torq"]

        if "MFCS" in self.df.columns and len(self.df["MFCS"]) >= 5:
            std5 = self.df["MFCS"].rolling(window=5).std().iloc[-1]
            if pd.notna(std5) and std5 > self.mfcs_fluctuation_threshold:
                status = 1

        if status == 1:
            return {"code": alert_code, "params": params}

    async def mfcs_sudden_spikes_alert(self):
        alert_code = "mfcs_spike"
        status = 0
        params = ["fdr_actrpm", "fhoff_actmpm", "meltpres", "fdr_torq"]

        if "MFCS" in self.df.columns and len(self.df["MFCS"]) >= 2:
            spike = self.df["MFCS"].diff().abs().iloc[-1]
            if pd.notna(spike) and spike > self.mfcs_spike_threshold:
                status = 1
                params.append(f"MFCS spike: {spike:.2f}")

        if status == 1:
            return {"code": alert_code, "params": params}

    async def mfcs_decreasing_trend_alert(self):
        alert_code = "mfcs_trend"
        status = 0
        params = ["fdr_actrpm", "fhoff_actmpm", "meltpres", "fdr_torq"]

        if "MFCS" in self.df.columns and len(self.df["MFCS"]) >= self.mfcs_trend_window:
            w = self.df["MFCS"].iloc[-self.mfcs_trend_window:].dropna()
            if len(w) == self.mfcs_trend_window:
                slope = _polyfit_slope(w)
                if slope is not None and slope < self.mfcs_trend_slope_threshold:
                    status = 1
                    params.append(f"MFCS trend slope: {slope:.2f}")

        if status == 1:
            return {"code": alert_code, "params": params}

    # ---------------------------- Alert Orchestrator -------------------------

    async def generate_alerts(self) -> List[Dict[str, Any]]:
        """
        Runs all alert functions and returns a list of alert dicts.
        Enable/disable individual alerts by commenting lines here if needed.
        """
        tasks = [
            self.zonewise_temperature_setpoint_deviation_alert(),
            self.zonewise_temperature_heater_stability_alert(),
            self.zonewise_temperature_spike_and_drop_alert(),

            self.screw_rotation_setpoint_deviation_alert(),
            self.screw_rotation_anomaly_detection_alert(),

            self.torque_anomaly_detection_alert(),
            self.torque_spike_detection_alert(),

            self.mfi_high_fluctuation_alert(),
            self.mfi_sudden_drops_alert(),
            self.mfi_gradual_rising_trend_alert(),

            self.shear_heat_spike_alert(),
            # self.shear_heat_threshold_alert(),  # uncomment if needed

            self.mfcs_fluctuation_alert(),
            self.mfcs_sudden_spikes_alert(),
            # self.mfcs_decreasing_trend_alert()  # optional
        ]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]


# ------------------------------ Example Runner -------------------------------
# This block is only for local quick checks. In production, the service calls
# ProcessParameterAnalysis(window, config=..., config_path=...) directly.

if __name__ == "__main__":
    import random

    async def _demo():
        # Simulated 60-row window
        rng = np.random.default_rng(0)
        win = []
        for i in range(60):
            row = {
                "timestamp": f"2025-02-12 12:00:{i:02d}",
                "ext_actrpm": float(rng.normal(100, 5)),
                "ext_setrpm": 100.0,
                "fdr_actrpm": float(rng.normal(30, 2)),
                "fdr_setrpm": 30.0,
                "bz1_actpv": float(rng.normal(200, 5)),
                "bz1_setpv": 200.0,
                "bz2_actpv": float(rng.normal(205, 5)),
                "bz2_setpv": 205.0,
                "ext_torq": float(rng.normal(50, 3)),
                "fdr_torq": float(rng.normal(20, 2)),
                "fhoff_torq": float(rng.normal(25, 2)),
                "melttemp": float(rng.normal(250, 5)),
                "meltpres": float(rng.normal(5, 1)),
                "fhoff_actmpm": float(rng.normal(60, 3))
            }
            win.append(row)

        # Minimal inline config (optional). If omitted, auto-detection kicks in.
        cfg = {
            "zones": {"bz": ["bz1", "bz2"]},
            "thresholds": {"setpoint_tolerance": 5, "stability_tolerance": 4, "spike_tolerance": 6, "z_score_threshold": 1},
            "mfi": {"C": 1.0, "n": 0.4, "fluctuation_threshold": 2.0, "drop_threshold": 2.0, "trend_window": 8, "trend_slope_threshold": 0.07},
            "shear_heat": {"spike_tolerance": 15.0, "high_threshold": 50.0},
            "mfcs": {"corr_window": 50, "fluctuation_threshold": 0.01, "spike_threshold": 0.02, "trend_window": 5, "trend_slope_threshold": -0.05}
        }

        analyzer = ProcessParameterAnalysis(win, config=cdf := cfg)
        alerts = await analyzer.generate_alerts()
        for a in alerts:
            print(a)

    asyncio.run(_demo())

"""
Changes in service.py one will have to make:
loading per machine config

import os, yaml

def load_machine_config(tenant_id: int, machine_id: int):
    # Example path scheme; adjust as needed.
    path = f"configs/extruder_{tenant_id}_{machine_id}.yaml"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return None

# ...
process_data = await fetch_process_data(api_key, machine_id, start_time, end_time)
cfg = load_machine_config(tenant_id, machine_id)
analyzer = ProcessParameterAnalysis(process_data, config=cfg)
process_alerts_data = await analyzer.generate_alerts()

"""
