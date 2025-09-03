import argparse
import logging
import os
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
DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"
# These weights are part of the model's logic and will be stored in the config
FEATURE_WEIGHTS = {
    'kurtosis': 0.25,
    'crest_factor': 0.2,
    'thd': 0.2,
    'sideband_energy_ratio': 0.2,
    'modulation_index': 0.15
}

# --- Core Feature Engineering Functions ---

def extract_freq_value_pairs(df_row, prefix, target_freq, tolerance):
    """Extracts amplitude values where frequencies are near the target fault frequency."""
    selected_values = []
    for axis in ['x', 'y', 'z']:
        for i in range(20):  # Assuming max 20 harmonics
            freq_col = f"{prefix}.{axis}.acceleration.frequencies[{i}]"
            value_col = f"{prefix}.{axis}.acceleration.values[{i}]"
            if freq_col in df_row and value_col in df_row and pd.notna(df_row[freq_col]):
                if abs(df_row[freq_col] - target_freq) <= tolerance:
                    selected_values.append(df_row[value_col])
    return np.array(selected_values)

def compute_features(signal_values):
    """Computes vibration-based statistical features from an array of amplitudes."""
    if len(signal_values) == 0:
        return {key: 0 for key in FEATURE_WEIGHTS.keys()}

    signal = np.array(signal_values)
    sk = kurtosis(signal, fisher=False)
    crest = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-6)
    
    fundamental = np.max(np.abs(signal))
    harmonics = np.delete(np.abs(signal), np.argmax(np.abs(signal)))
    thd = np.sqrt(np.sum(harmonics**2)) / (fundamental + 1e-6)

    peaks, _ = find_peaks(np.abs(signal))
    sideband_energy = np.sum(np.abs(signal[peaks])**2) if len(peaks) > 0 else 0
    ser = sideband_energy / (np.sum(signal**2) + 1e-6)

    smi = np.std(np.diff(signal[peaks])) / (np.mean(np.abs(signal)) + 1e-6) if len(peaks) >= 2 else 0

    return {'kurtosis': sk, 'crest_factor': crest, 'thd': thd, 'sideband_energy_ratio': ser, 'modulation_index': smi}

def compute_bdf(features, weights):
    """Computes the Bearing Degradation Factor (BDF) from weighted features."""
    return sum(features[key] * weights.get(key, 0) for key in features)

# --- Main Execution Logic ---

def main(args):
    """Main function to generate bearing fault configs and log to MLflow."""
    if mlflow.active_run():
        mlflow.end_run()

    experiment_name = f"vibration_bearing_fault_monitoring/{args.tenant_id}/{args.machine_id}"
    mlflow_wrapper = MLflowBase(experiment_name=experiment_name)
    
    with mlflow_wrapper.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        logger.info(f"🚀 Started MLflow Run: {run_id} in Experiment: '{experiment_name}'")

        mlflow_wrapper.log_params({
            "tenant_id": args.tenant_id, "machine_id": args.machine_id,
            "dataset_filename": args.dataset_filename, "run_name": args.run_name,
            "bpfo_hz": args.bpfo, "bpfi_hz": args.bpfi,
            "frequency_tolerance_hz": args.tolerance, 
            "bdf_threshold_percentile": args.percentile
        })

        dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        logger.info(f"💾 Loading data from: {dataset_path}")
        df = pd.read_csv(dataset_path)

        results = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating BDF"):
            values_bpfo = extract_freq_value_pairs(row, 'motor', args.bpfo, args.tolerance)
            values_bpfi = extract_freq_value_pairs(row, 'motor', args.bpfi, args.tolerance)
            
            features_bpfo = compute_features(values_bpfo)
            features_bpfi = compute_features(values_bpfi)

            bdf_bpfo = compute_bdf(features_bpfo, FEATURE_WEIGHTS)
            bdf_bpfi = compute_bdf(features_bpfi, FEATURE_WEIGHTS)

            final_bdf = 0.5 * bdf_bpfo + 0.5 * bdf_bpfi
            results.append({'final_bdf': final_bdf})

        results_df = pd.DataFrame(results)

        # Statistically derive the threshold from the dataset's BDF distribution
        threshold = float(results_df['final_bdf'].quantile(args.percentile / 100.0))
        logger.info(f"Derived BDF threshold at {args.percentile}th percentile: {threshold:.4f}")

        fault_flags = results_df['final_bdf'] > threshold
        metrics = {
            "rows_processed": float(len(df)),
            "bdf_mean": float(results_df['final_bdf'].mean()),
            "bdf_std": float(results_df['final_bdf'].std()),
            "bdf_p99": float(results_df['final_bdf'].quantile(0.99)),
            "derived_bdf_threshold": threshold,
            "fault_rate_at_threshold_pct": float(fault_flags.mean() * 100)
        }
        mlflow_wrapper.log_metrics(metrics)

        # Build, save, and log the configuration artifact
        config = {
            "tenant_id": args.tenant_id, "machine_id": args.machine_id,
            "bpfo_hz": args.bpfo, "bpfi_hz": args.bpfi,
            "frequency_tolerance_hz": args.tolerance,
            "bdf_threshold": threshold,
            "feature_weights": FEATURE_WEIGHTS
        }
        
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        config_filename = f"vibration_bearing_config_{args.machine_id}.yaml"
        config_path = os.path.join(ARTIFACTS_DIR, config_filename)
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        
        mlflow_wrapper.log_artifact(run_id, config_path)
        logger.info(f"📜 Configuration saved and logged as {config_filename}")
        logger.info("🏁 Run finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Vibration-Based Bearing Fault configs for MLflow.")

    # example usage:
    # python vibration_bearing_model_training.py --tenant-id "your_tenant_id" --machine-id "your_machine_id" --dataset-filename "iotts.vibration.csv" --bpfo 133.1 --bpfi 106.9
    # tenant is 27, machine is 243 and dataset is iotts.vibration_243.csv
    # Required Arguments
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant ID.")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID.")
    parser.add_argument("--dataset-filename", type=str, required=True, help=f"Dataset CSV in '{DATA_DIR}/'.")
    parser.add_argument("--bpfo", type=float, required=True, help="Ball Pass Frequency Outer race (Hz).")
    parser.add_argument("--bpfi", type=float, required=True, help="Ball Pass Frequency Inner race (Hz).")

    # Optional Arguments
    parser.add_argument("--run-name", type=str, default="vibration_bdf_v1", help="Name for the MLflow run.")
    parser.add_argument("--tolerance", type=float, default=20.0, help="Frequency tolerance window in Hz.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Percentile to set the BDF threshold.")
    
    cli_args = parser.parse_args()
    main(cli_args)