import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import logging
import warnings

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow_base import MLflowBase
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


# --- Core Feature Engineering Functions ---

def calculate_rotor_unbalance_index(row, f_rot, tolerance):
    """Calculates the rotor unbalance index from a single row of FFT data."""
    axis_results = {}
    
    for axis in ['x', 'y', 'z']:
        amp_1x = None
        other_amps = []

        for i in range(20):  # Assuming max 20 harmonics
            freq_col = f"motor.{axis}.acceleration.frequencies[{i}]"
            value_col = f"motor.{axis}.acceleration.values[{i}]"

            if freq_col in row and value_col in row and pd.notna(row[freq_col]):
                freq = row[freq_col]
                amp = row[value_col]
                
                if abs(freq - f_rot) <= tolerance:
                    amp_1x = amp
                else:
                    other_amps.append(amp)

        if amp_1x is not None and len(other_amps) > 0:
            mean_others = np.mean(np.abs(other_amps)) + 1e-6
            imbalance_index = abs(amp_1x) / mean_others
            axis_results[axis] = imbalance_index

    return max(axis_results.values()) if axis_results else 0

# --- Main Execution Logic ---

def main(args):
    """Main function to generate rotor fault configs and log to MLflow."""
    if mlflow.active_run():
        mlflow.end_run()

    experiment_name = f"vibration_rotor_fault_monitoring/{args.tenant_id}/{args.machine_id}"
    mlflow_wrapper = MLflowBase(experiment_name=experiment_name)
    
    with mlflow_wrapper.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        logger.info(f"🚀 Started MLflow Run: {run_id} in Experiment: '{experiment_name}'")

        mlflow_wrapper.log_params({
            "tenant_id": args.tenant_id, "machine_id": args.machine_id,
            "dataset_filename": args.dataset_filename, "run_name": args.run_name,
            "rpm": args.rpm, "frequency_tolerance_hz": args.tolerance,
            "threshold_percentile_mild": args.p_mild,
            "threshold_percentile_significant": args.p_sig,
            "threshold_percentile_severe": args.p_sev,
        })

        dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        logger.info(f"💾 Loading data from: {dataset_path}")
        df = pd.read_csv(dataset_path)

        f_rot = args.rpm / 60.0
        results = [
            calculate_rotor_unbalance_index(row, f_rot, args.tolerance)
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating Unbalance Index")
        ]
        results_df = pd.DataFrame(results, columns=['unbalance_index'])

        # Statistically derive thresholds from the dataset's distribution
        mild_thresh = float(results_df['unbalance_index'].quantile(args.p_mild / 100.0))
        sig_thresh = float(results_df['unbalance_index'].quantile(args.p_sig / 100.0))
        sev_thresh = float(results_df['unbalance_index'].quantile(args.p_sev / 100.0))
        logger.info(f"Derived Thresholds: Mild > {mild_thresh:.2f}, Significant > {sig_thresh:.2f}, Severe > {sev_thresh:.2f}")

        metrics = {
            "rows_processed": float(len(df)),
            "unbalance_index_mean": float(results_df['unbalance_index'].mean()),
            "unbalance_index_std": float(results_df['unbalance_index'].std()),
            "unbalance_index_p99": float(results_df['unbalance_index'].quantile(0.99)),
            "derived_mild_threshold": mild_thresh,
            "derived_significant_threshold": sig_thresh,
            "derived_severe_threshold": sev_thresh,
        }
        mlflow_wrapper.log_metrics(metrics)

        # Build, save, and log the configuration artifact
        config = {
            "tenant_id": args.tenant_id, "machine_id": args.machine_id,
            "rpm": args.rpm, "frequency_tolerance_hz": args.tolerance,
            "unbalance_threshold_mild": mild_thresh,
            "unbalance_threshold_significant": sig_thresh,
            "unbalance_threshold_severe": sev_thresh,
        }
        
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        config_filename = f"vibration_rotor_config_{args.machine_id}.yaml"
        config_path = os.path.join(ARTIFACTS_DIR, config_filename)
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        
        mlflow_wrapper.log_artifact(run_id, config_path)
        logger.info(f"📜 Configuration saved and logged as {config_filename}")
        logger.info("🏁 Run finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Vibration-Based Rotor Fault configs for MLflow.")

    # example usage:
    # python vibration_rotor_model_training.py --tenant-id "27" --machine-id "243" --dataset-filename "iotts.vibration_243.csv" --rpm 1800

    # Required Arguments
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant ID.")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID.")
    parser.add_argument("--dataset-filename", type=str, required=True, help=f"Dataset CSV in '{DATA_DIR}/'.")
    parser.add_argument("--rpm", type=float, required=True, help="Nominal shaft speed of the motor in RPM.")

    # Optional Arguments
    parser.add_argument("--run-name", type=str, default="vibration_rotor_unbalance_v1", help="Name for the MLflow run.")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Frequency tolerance window in Hz for finding 1x RPM peak.")
    parser.add_argument("--p-mild", type=float, default=95.0, help="Percentile to set the 'Mild' unbalance threshold.")
    parser.add_argument("--p-sig", type=float, default=99.0, help="Percentile to set the 'Significant' unbalance threshold.")
    parser.add_argument("--p-sev", type=float, default=99.8, help="Percentile to set the 'Severe' unbalance threshold.")
    
    cli_args = parser.parse_args()
    main(cli_args)