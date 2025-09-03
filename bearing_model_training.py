# generate_config.py

import argparse
import logging
import math
import os
import re
import warnings
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow_base import MLflowBase
from scipy import stats
from tqdm import tqdm

# --- Pre-Script Configuration & Constants ---

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for input data
DATA_DIR = "data"
# Directory for local artifact output
ARTIFACTS_DIR = "artifacts"

# Feature weights (kept as a constant for simplicity for non-ML users)
# To change this, a developer would edit the script directly.
WEIGHTS = {
    "spectral_kurtosis_rms_0": 0.20,
    "spectral_crest_factor_rms_1": 0.10,
    "total_harmonic_distortion_rms_2": 0.30,
    "sideband_energy_ratio_rms_3": 0.10,
    "sideband_modulation_index_rms_4": 0.10,
    "bpfo_band_energy": 0.10,
    "bpfi_band_energy": 0.10,
}

# Robust normalization percentile bounds (avoid outliers)
LOW_PCT, HIGH_PCT = 1.0, 99.0

# --- Helper Functions (from Notebook) ---

CH_PAT = re.compile(r"^(ch[123])_(\d+)$")

def discover_available_bins(df: pd.DataFrame) -> List[int]:
    """Return sorted list of bin indices k that exist for ALL channels ch1_k, ch2_k, ch3_k."""
    ch_bins = {f"ch{i}": set() for i in [1, 2, 3]}
    for col in df.columns:
        m = CH_PAT.match(col)
        if not m:
            continue
        ch, k = m.group(1), int(m.group(2))
        ch_bins[ch].add(k)
    common = sorted(list(ch_bins["ch1"] & ch_bins["ch2"] & ch_bins["ch3"]))
    if not common:
        raise ValueError("No common frequency bins found across ch1/ch2/ch3.")
    return common

def calculate_rms_dynamic(df: pd.DataFrame, bins: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    """Compute RMS across 3 channels for the provided bin indices."""
    rms_cols = []
    for k in bins:
        chs = [f"ch1_{k}", f"ch2_{k}", f"ch3_{k}"]
        if not all(c in df.columns for c in chs):
            continue
        out_col = f"rms_{k}"
        df[out_col] = np.sqrt(df[chs].pow(2).sum(axis=1) / 3.0)
        rms_cols.append(out_col)
    if not rms_cols:
        raise ValueError("No RMS columns created — check your input data.")
    return df, rms_cols

def nearest_bin(target_k: int, bins: List[int]) -> int:
    """Pick the available bin index closest to target_k."""
    idx = np.searchsorted(bins, target_k)
    candidates = []
    if idx < len(bins):
        candidates.append(bins[idx])
    if idx > 0:
        candidates.append(bins[idx - 1])
    return min(candidates, key=lambda x: abs(x - target_k))

def neighbor_triplet(center_k: int, bins: List[int]) -> List[int]:
    """Return [lower, center, upper] bin indices from available bins."""
    c = nearest_bin(center_k, bins) if center_k not in bins else center_k
    pos = bins.index(c)
    if pos == 0:
        return [bins[0], bins[0], bins[min(1, len(bins) - 1)]]
    if pos == len(bins) - 1:
        return [bins[max(len(bins) - 2, 0)], bins[-1], bins[-1]]
    return [bins[pos - 1], bins[pos], bins[pos + 1]]

def band_energy_around_dynamic(df_rms: pd.DataFrame, center_freq_hz: float, freq_res: float, bins: List[int]) -> pd.Series:
    """Mean of available neighbor triplet around nearest bin to center_freq_hz."""
    target_k = int(round(center_freq_hz / freq_res))
    trip = neighbor_triplet(target_k, bins)
    cols = [f"rms_{k}" for k in trip if f"rms_{k}" in df_rms.columns]
    return df_rms[cols].mean(axis=1)

def spectral_kurtosis(full_spectrum: np.ndarray) -> float:
    return float(stats.kurtosis(full_spectrum, fisher=True, bias=False, nan_policy="omit"))

def spectral_crest_factor(full_spectrum: np.ndarray) -> float:
    full_spectrum = np.asarray(full_spectrum, dtype=float)
    peak = np.nanmax(np.abs(full_spectrum))
    rms = np.sqrt(np.nanmean(full_spectrum ** 2))
    return float(peak / rms) if rms > 0 else np.nan

def thd_centered_triplet(triplet_vals: np.ndarray) -> float:
    if triplet_vals is None or len(triplet_vals) != 3:
        return np.nan
    fundamental = float(triplet_vals[1])
    side = math.sqrt(float(triplet_vals[0]) ** 2 + float(triplet_vals[2]) ** 2)
    return float(side / fundamental) if fundamental != 0 else np.nan

def sideband_energy_ratio(triplet_vals: np.ndarray) -> float:
    if triplet_vals is None or len(triplet_vals) != 3:
        return np.nan
    c = float(triplet_vals[1])
    s = math.sqrt(float(triplet_vals[0]) ** 2 + float(triplet_vals[2]) ** 2)
    return float(s / c) if c != 0 else np.nan

def sideband_modulation_index(triplet_vals: np.ndarray) -> float:
    if triplet_vals is None or len(triplet_vals) != 3:
        return np.nan
    c = float(triplet_vals[1])
    v = float(triplet_vals[0]) + float(triplet_vals[2])
    return float(v / c) if c != 0 else np.nan

def bpfo_bpfi_from_geometry(cfg: Dict) -> Tuple[float, float]:
    d = cfg["ball_diameter"]
    D = cfg["pitch_diameter"]
    n = cfg["num_rolling_elements"]
    contact_angle = np.deg2rad(cfg["contact_angle_deg"])
    fr = cfg["shaft_rpm"] / 60.0
    bpfo = fr * n / 2.0 * (1.0 - (d / D) * np.cos(contact_angle))
    bpfi = fr * n / 2.0 * (1.0 + (d / D) * np.cos(contact_angle))
    return float(bpfo), float(bpfi)

def robust_minmax(series: pd.Series, low=LOW_PCT, high=HIGH_PCT) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"min": 0.0, "max": 1.0}
    lo = float(np.percentile(s, low))
    hi = float(np.percentile(s, high))
    if hi <= lo:
        lo = float(np.nanmin(s))
        hi = float(np.nanmax(s))
        if hi <= lo:
            return {"min": float(lo), "max": float(lo + 1e-9)}
    return {"min": lo, "max": hi}

def normalize_value_minmax(x: float, stats: Dict[str, float]) -> float:
    vmin, vmax = stats.get("min"), stats.get("max")
    if vmax is None or vmin is None or vmax == vmin:
        return 0.0
    return float((x - vmin) / (vmax - vmin))

def compute_bdf_row(row: pd.Series, weights: Dict[str, float], norm_cfg: Dict[str, Dict[str, float]]) -> float:
    total, wsum = 0.0, 0.0
    for feat, w in weights.items():
        if feat in row.index:
            x = float(row[feat])
            xnorm = normalize_value_minmax(x, norm_cfg.get(feat, {"min": 0.0, "max": 1.0}))
            total += xnorm * w
            wsum += w
    return float(total / wsum) if wsum > 0 else 0.0

# --- Main Execution Logic ---

def main(args):
    """
    Main function to run the config generation and MLflow logging pipeline.
    """
    if mlflow.active_run():
        mlflow.end_run()

    # 1. MLflow Setup
    experiment_name = f"bearing_fault_monitoring/{args.tenant_id}/{args.machine_id}"
    mlbase = MLflowBase(experiment_name)
    run = mlbase.start_run(run_name=args.approach)
    run_id = run.info.run_id
    mlflow.set_tag("run_tag", args.run_tag)
    logger.info(f"🚀 Started MLflow Run: {run_id} in Experiment: '{experiment_name}'")

    try:
        # 2. Log Initial Parameters
        mlbase.log_params({
            "tenant_id": args.tenant_id,
            "machine_id": args.machine_id,
            "approach": args.approach,
            "dataset_filename": args.dataset_filename,
            "freq_resolution": args.freq_resolution,
            "bdf_threshold": args.bdf_threshold,
            "shaft_rpm": args.shaft_rpm
        })

        # 3. Load Data
        dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        logger.info(f"💾 Loading data from: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please place it in the '{DATA_DIR}' folder.")
        df = pd.read_csv(dataset_path)

        # 4. Feature Engineering
        logger.info("🛠️  Starting feature engineering...")
        available_bins = discover_available_bins(df)
        df, rms_cols = calculate_rms_dynamic(df, available_bins)
        
        bearing_config = {
            "shaft_rpm": args.shaft_rpm,
            "ball_diameter": args.ball_diameter,
            "pitch_diameter": args.pitch_diameter,
            "num_rolling_elements": args.num_elements,
            "contact_angle_deg": args.contact_angle
        }
        bpfo_hz, bpfi_hz = bpfo_bpfi_from_geometry(bearing_config)

        fr_hz = bearing_config["shaft_rpm"] / 60.0
        target_bin = int(round(fr_hz / args.freq_resolution))
        fundamental_bin = nearest_bin(target_bin, available_bins)

        df["bpfo_band_energy"] = band_energy_around_dynamic(df[rms_cols], bpfo_hz, args.freq_resolution, available_bins)
        df["bpfi_band_energy"] = band_energy_around_dynamic(df[rms_cols], bpfi_hz, args.freq_resolution, available_bins)

        ordered_rms_cols = [f"rms_{k}" for k in available_bins]
        rms_arr = df[ordered_rms_cols].values
        
        trip_bins = neighbor_triplet(fundamental_bin, available_bins)
        trip_cols = [f"rms_{k}" for k in trip_bins]

        for i in tqdm(range(len(df)), desc="Calculating stateless features"):
            row_spec = rms_arr[i, :]
            trip_vals = df.loc[df.index[i], trip_cols].values.astype(float)
            df.at[i, "spectral_kurtosis_rms_0"] = spectral_kurtosis(row_spec)
            df.at[i, "spectral_crest_factor_rms_1"] = spectral_crest_factor(row_spec)
            df.at[i, "total_harmonic_distortion_rms_2"] = thd_centered_triplet(trip_vals)
            df.at[i, "sideband_energy_ratio_rms_3"] = sideband_energy_ratio(trip_vals)
            df.at[i, "sideband_modulation_index_rms_4"] = sideband_modulation_index(trip_vals)
        logger.info("✅ Feature engineering complete.")

        # 5. Build Normalization Stats
        logger.info("📊 Building robust min-max normalization stats...")
        all_feature_keys = list(WEIGHTS.keys())
        norm_minmax = {key: robust_minmax(df[key]) for key in all_feature_keys if key in df.columns}
        
        # 6. Preview BDF Distribution
        bdf_preview = df[all_feature_keys].apply(
            lambda r: compute_bdf_row(r, WEIGHTS, norm_minmax), axis=1
        )
        logger.info("📈 BDF Preview Stats:\n" + str(bdf_preview.describe(percentiles=[0.5, 0.75, 0.9, 0.95])))

        # 7. Build and Log YAML Config
        final_config = {
            "shaft_rpm": float(bearing_config["shaft_rpm"]),
            "ball_diameter": float(bearing_config["ball_diameter"]),
            "pitch_diameter": float(bearing_config["pitch_diameter"]),
            "num_rolling_elements": int(bearing_config["num_rolling_elements"]),
            "contact_angle_deg": float(bearing_config["contact_angle_deg"]),
            "freq_resolution": float(args.freq_resolution),
            "bdf_threshold": float(args.bdf_threshold),
            "weights": {k: float(v) for k, v in WEIGHTS.items()},
            "normalization": {
                "minmax": {k: {"min": float(v["min"]), "max": float(v["max"])} for k, v in norm_minmax.items()}
            }
        }
        
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        yaml_path = os.path.join(ARTIFACTS_DIR, "bearing_config.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(final_config, f, sort_keys=False, default_flow_style=False)
        
        mlbase.log_artifact(run_id=run_id, local_path=yaml_path)
        logger.info(f"📜 Saved and logged config to MLflow: {yaml_path}")
        
        # 8. Log Summary Metrics
        metrics = {
            "rows_processed": float(len(df)),
            "bdf_p50": float(np.nanpercentile(bdf_preview, 50)),
            "bdf_p75": float(np.nanpercentile(bdf_preview, 75)),
            "bdf_p90": float(np.nanpercentile(bdf_preview, 90)),
            "bdf_mean": float(np.nanmean(bdf_preview)),
            "bdf_std": float(np.nanstd(bdf_preview)),
            "bpfo_hz": float(bpfo_hz),
            "bpfi_hz": float(bpfi_hz),
            "fundamental_bin_guess": float(fundamental_bin),
        }
        mlbase.log_metrics(metrics)
        logger.info("📝 Logged final metrics to MLflow.")

    except Exception as e:
        logger.error(f"❌ An error occurred during the run: {e}")
        raise
    finally:
        # 9. End MLflow Run
        mlflow.end_run()
        logger.info("🏁 Run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and log a bearing fault detection config to MLflow.")

    # example usage:
    # python bearing_model_training.py --tenant-id "28" --machine-id "257" --dataset-filename "iotts.harmonics_257.csv" --shaft-rpm 1480 --ball-diameter 8.0 --pitch-diameter 40.0 --num-elements 8
    # example production run:
    # python bearing_model_training.py --tenant-id "28" --machine-id "257" --dataset-filename "iotts.harmonics_257.csv" --shaft-rpm 1480 --ball-diameter 8.0 --pitch-diameter 40.0 --num-elements 8 --run-tag @production
    # example testing run:
    # python bearing_model_training.py --tenant-id "28" --machine-id "257" --dataset-filename "iotts.harmonics_257.csv" --shaft-rpm 1480 --ball-diameter 8.0 --pitch-diameter 40.0 --num-elements 8 --run-tag @testing
    
    # --- Required Arguments ---
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant ID for the machine.")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID for the new equipment.")
    parser.add_argument("--dataset-filename", type=str, required=True, help=f"Filename of the CSV dataset (must be in the '{DATA_DIR}/' folder).")
    
    # --- Bearing Geometry Arguments ---
    parser.add_argument("--shaft-rpm", type=float, required=True, help="Shaft speed in RPM.")
    parser.add_argument("--ball-diameter", type=float, required=True, help="Diameter of a single rolling element (ball).")
    parser.add_argument("--pitch-diameter", type=float, required=True, help="Pitch diameter of the bearing.")
    parser.add_argument("--num-elements", type=int, required=True, help="Number of rolling elements in the bearing.")
    parser.add_argument("--contact-angle", type=float, default=0.0, help="Contact angle in degrees (default: 0.0).")

    # --- Optional Model/Knob Arguments ---
    parser.add_argument("--approach", type=str, default="bdf_stateless_v1", help="Name for the approach/run (default: bdf_stateless_v1).")
    parser.add_argument("--freq-resolution", type=float, default=50.0, help="Frequency resolution in Hz per bin (default: 50.0).")
    parser.add_argument("--bdf-threshold", type=float, default=0.9, help="Default BDF decision threshold (default: 0.7).")

    parser.add_argument("--run-tag", type=str, choices=["@production", "@testing"], default="@testing", help="Tag for the MLflow run: '@production' or '@testing'.")

    cli_args = parser.parse_args()
    main(cli_args)