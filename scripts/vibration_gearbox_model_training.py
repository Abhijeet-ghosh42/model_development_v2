import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
import logging
import warnings

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow_base import MLflowBase
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from tqdm import tqdm

# --- Pre-Script Configuration & Constants ---

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# DATA_DIR = "data"
# ARTIFACTS_DIR = "artifacts"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# These weights are based on domain knowledge and are part of the config
FEATURE_WEIGHTS = {
    'kurtosis': 0.2,
    'crest_factor': 0.2,
    'sideband_energy_ratio': 0.2,
    'modulation_index': 0.15,
    'entropy': 0.15,
    'gmf_energy_ratio': 0.1
}

# --- Core Feature Engineering Functions (from your inference script) ---

def extract_freq_value_pairs(df_row, target_freq, tolerance):
    """Extracts amplitude values where frequencies are near the target fault frequency."""
    selected_values = []
    prefix = 'gearbox'
    for axis in ['x', 'y', 'z']:
        for i in range(20):
            freq_col = f"{prefix}.{axis}.acceleration.frequencies[{i}]"
            value_col = f"{prefix}.{axis}.acceleration.values[{i}]"
            if freq_col in df_row and value_col in df_row and pd.notna(df_row[freq_col]):
                if abs(df_row[freq_col] - target_freq) <= tolerance:
                    selected_values.append(df_row[value_col])
    return np.array(selected_values)

def compute_gearbox_features(signal_values):
    """Computes vibration-based statistical features from an array of amplitudes."""
    if len(signal_values) == 0:
        return {key: 0 for key in FEATURE_WEIGHTS.keys()}

    signal = np.array(signal_values)
    # Z-score normalization for stability
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    sk = kurtosis(signal, fisher=False)
    crest = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-6)
    
    fundamental = np.max(np.abs(signal))
    harmonics = np.delete(np.abs(signal), np.argmax(np.abs(signal)))
    thd = np.sqrt(np.sum(harmonics**2)) / (fundamental + 1e-6)

    peaks, _ = find_peaks(np.abs(signal))
    sideband_energy = np.sum(np.abs(signal[peaks])**2) if len(peaks) > 0 else 0
    ser = sideband_energy / (np.sum(signal**2) + 1e-6)

    smi = np.std(np.diff(signal[peaks])) / (np.mean(np.abs(signal)) + 1e-6) if len(peaks) >= 2 else 0
    
    p = np.abs(signal)**2
    p_norm = p / (np.sum(p) + 1e-6)
    entropy = -np.sum(p_norm * np.log(p_norm + 1e-6))

    top2 = np.sort(np.abs(signal))[-2:]
    gmf_ratio = top2[0] / (top2[1] + 1e-6) if len(top2) == 2 else 0
    
    return {
        'kurtosis': sk, 'crest_factor': crest, 'thd': thd,
        'sideband_energy_ratio': ser, 'modulation_index': smi,
        'entropy': entropy, 'gmf_energy_ratio': gmf_ratio
    }

def compute_gdf(features, weights):
    """Computes the Gear Degradation Factor (GDF) from weighted features."""
    return sum(features[key] * weights.get(key, 0) for key in features)

# --- Main Execution Logic ---

def main(args):
    """Main function to generate gearbox fault configs and log to MLflow."""
    if mlflow.active_run():
        mlflow.end_run()

    experiment_name = f"vibration_gearbox_fault_monitoring/{args.tenant_id}/{args.machine_id}"
    mlflow_wrapper = MLflowBase(experiment_name=experiment_name)
    
    with mlflow_wrapper.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        logger.info(f"🚀 Started MLflow Run: {run_id} in Experiment: '{experiment_name}'")

        mlflow_wrapper.log_params({
            "tenant_id": args.tenant_id, "machine_id": args.machine_id,
            "dataset_filename": args.dataset_filename, "run_name": args.run_name,
            "gmf_hz": args.gmf, "frequency_tolerance_hz": args.tolerance,
            "gdf_threshold_percentile": args.percentile
        })

        dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        logger.info(f"💾 Loading data from: {dataset_path}")
        df = pd.read_csv(dataset_path)

        results = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating GDF"):
            values = extract_freq_value_pairs(row, args.gmf, args.tolerance)
            features = compute_gearbox_features(values)
            gdf = compute_gdf(features, FEATURE_WEIGHTS)
            results.append({'gdf': gdf})
        results_df = pd.DataFrame(results)

        # Derive threshold from data distribution
        threshold = float(results_df['gdf'].quantile(args.percentile / 100.0))
        logger.info(f"Derived GDF threshold at {args.percentile}th percentile: {threshold:.4f}")

        fault_flags = results_df['gdf'] > threshold
        metrics = {
            "rows_processed": float(len(df)),
            "gdf_mean": float(results_df['gdf'].mean()),
            "gdf_std": float(results_df['gdf'].std()),
            "gdf_p99": float(results_df['gdf'].quantile(0.99)),
            "gdf_max": float(results_df['gdf'].max()),
            "derived_gdf_threshold": threshold,
            "fault_rate_at_threshold_pct": float(fault_flags.mean() * 100)
        }
        mlflow_wrapper.log_metrics(metrics)

        # Build, save, and log the configuration artifact
        config = {
            "tenant_id": args.tenant_id, "machine_id": args.machine_id,
            "gmf_hz": args.gmf, "frequency_tolerance_hz": args.tolerance,
            "gdf_threshold": threshold, "feature_weights": FEATURE_WEIGHTS
        }
        
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        config_filename = f"vibration_gearbox_config_{args.machine_id}.yaml"
        config_path = os.path.join(ARTIFACTS_DIR, config_filename)
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        
        mlflow_wrapper.log_artifact(run_id, config_path)
        logger.info(f"📜 Configuration saved and logged as {config_filename}")
        logger.info("🏁 Run finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Vibration-Based Gearbox Fault configs.")

    # example usage:
    # python vibration_gearbox_model_training.py --tenant-id "27" --machine-id "243" --dataset-filename "iotts.vibration_243.csv" --gmf 1080
    
    # Required Arguments
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant ID.")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID.")
    parser.add_argument("--dataset-filename", type=str, required=True, help=f"Dataset CSV in '{DATA_DIR}/'.")
    parser.add_argument("--gmf", type=float, required=True, help="Gear Meshing Frequency (Hz).")

    # Optional Arguments
    parser.add_argument("--run-name", type=str, default="vibration_gdf_v1", help="Name for the MLflow run.")
    parser.add_argument("--tolerance", type=float, default=100.0, help="Frequency tolerance window in Hz.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Percentile to set the GDF threshold.")
    
    cli_args = parser.parse_args()
    main(cli_args)