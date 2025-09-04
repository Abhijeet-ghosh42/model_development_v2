import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import logging
import math
import warnings

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow_base import MLflowBase

from data_loaders import load_esa_harmonics_api, load_local_csv


# --- Pre-Script Configuration & Constants ---

# Suppress warnings and setup basic logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# # Directory for input data
# DATA_DIR = "data"
# # Directory for local artifact output
# ARTIFACTS_DIR = "artifacts"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")


# --- Helper Functions ---

def calculate_imbalance(row: pd.Series) -> float:
    """Calculates the difference between the max and min values in a row."""
    return np.max(row) - np.min(row)

def generate_run_notes(params: dict, metrics: dict, output_yaml: str) -> str:
    """Generates a markdown note for the MLflow run."""
    def _fmt(x, nd=4):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return "NaN"
        try: return f"{float(x):.{nd}f}"
        except: return str(x)

    return f"""
### Run Documentation

**Deliverable:**
- `{output_yaml}` — contains thresholds for detecting winding faults in real-time.

---

**Parameters:**
- `tenant_id`: {params['tenant_id']}
- `machine_id`: {params['machine_id']}
- `dataset_filename`: {params['dataset_filename']}

---

**Metrics (for benchmarking):**
- **quality_index**: **{_fmt(metrics.get('quality_index', 'N/A'))}**
  - Higher score suggests thresholds derived from a larger, more stable dataset.
- `cur_imbalance_mean/std`: {_fmt(metrics.get('cur_imbalance_mean', 'N/A'))} / {_fmt(metrics.get('cur_imbalance_std', 'N/A'))}
  - Captures imbalance between current phases.
- `vol_imbalance_mean/std`: {_fmt(metrics.get('vol_imbalance_mean', 'N/A'))} / {_fmt(metrics.get('vol_imbalance_std', 'N/A'))}
  - Captures imbalance between voltage phases.
- `current_threshold`: {_fmt(metrics.get('current_threshold', 'N/A'))}
- `voltage_threshold`: {_fmt(metrics.get('voltage_threshold', 'N/A'))}

---

**Decision Context:**
The YAML config (artifact) is the actual model consumed by the online detector. Metrics provide internal validation and comparability across runs. Promotion/selection is based on `quality_index`.
"""

# --- Main Execution Logic ---

def main(args):
    """
    Main function to run the winding fault config generation and MLflow logging pipeline.
    """
    if mlflow.active_run():
        mlflow.end_run()

    # 1. MLflow Setup
    experiment_name = f"winding_fault_monitoring/{args.tenant_id}/{args.machine_id}"
    mlbase = MLflowBase(experiment_name)
    run = mlbase.start_run(run_name=args.run_name)
    run_id = run.info.run_id
    mlflow.set_tag("run_tag", args.run_tag)
    logger.info(f"🚀 Started MLflow Run: {run_id} in Experiment: '{experiment_name}'")

    try:
        # 2. Log Initial Parameters
        params_to_log = {
            "run_name": args.run_name,
            "dataset_filename": args.dataset_filename,
            "tenant_id": args.tenant_id,
            "machine_id": args.machine_id,
        }
        mlbase.log_params(params_to_log)

        # 3. Load & Prepare Data
        # dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        # logger.info(f"💾 Loading data from: {dataset_path}")
        # if not os.path.exists(dataset_path):
        #     raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please place it in the '{DATA_DIR}' folder.")
        
        # df = pd.read_csv(dataset_path)

        if args.source == "local":
            dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
            logger.info(f"💾 Loading data (local): {dataset_path}")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please place it in the '{DATA_DIR}' folder.")
            df = load_local_csv(dataset_path)
        else:
            # API mode
            if not all([args.api_base_url, args.api_key, args.start_datetime_utc, args.end_datetime_utc]):
                raise ValueError("For --source api, provide --api-base-url, --api-key, --start-datetime-utc, and --end-datetime-utc")

            logger.info(
                f"🌐 Loading data (API): {args.api_base_url} | machine_id={args.machine_id} | "
                f"{args.start_datetime_utc} → {args.end_datetime_utc}"
            )
            df = load_esa_harmonics_api(
                base_url=args.api_base_url,
                api_key=args.api_key,
                machine_id=args.machine_id,
                start_utc=args.start_datetime_utc,
                end_utc=args.end_datetime_utc,
                endpoint="/api/v1/energy/",
                max_span_hours=args.max_span_hours,
                extra_params={"response_type": "raw"},
            )
            if df.empty:
                raise ValueError("API returned no data for the requested time range.")


        drop_cols = [c for c in ["_id", "metaData.tenant_id", "metaData.machine_id"] if c in df.columns]
        if drop_cols: df = df.drop(columns=drop_cols, errors="ignore")
        mlbase.log_metrics({"n_rows": float(df.shape[0]), "n_cols": float(df.shape[1])})

        # 4. Compute Phase Imbalance
        logger.info("🛠️  Calculating phase imbalance...")
        current_cols = [c for c in df.columns if c.startswith("c")]
        voltage_cols = [c for c in df.columns if c.startswith("v")]

        current_imbalance = df[current_cols].apply(calculate_imbalance, axis=1)
        voltage_imbalance = df[voltage_cols].apply(calculate_imbalance, axis=1)

        cur_mean, cur_std = current_imbalance.mean(), current_imbalance.std()
        vol_mean, vol_std = voltage_imbalance.mean(), voltage_imbalance.std()
        
        imbalance_metrics = {
            "cur_imbalance_mean": cur_mean, "cur_imbalance_std": cur_std,
            "vol_imbalance_mean": vol_mean, "vol_imbalance_std": vol_std
        }
        mlbase.log_metrics(imbalance_metrics)
        logger.info(f"Current imbalance → mean {cur_mean:.3f}, std {cur_std:.3f}")
        logger.info(f"Voltage imbalance → mean {vol_mean:.3f}, std {vol_std:.3f}")

        # 5. Compute Thresholds
        logger.info("📊 Deriving thresholds...")
        cur_thr = cur_mean + 3 * cur_std
        vol_thr = vol_mean + 3 * vol_std
        harmonic_sums = df[current_cols + voltage_cols].sum(axis=1)
        harm_thr = harmonic_sums.mean() + 3 * harmonic_sums.std()
        stat_thr = df[current_cols + voltage_cols].stack().mean() + 3 * df[current_cols + voltage_cols].stack().std()

        thresholds = {
            "current_threshold": round(cur_thr, 3),
            "voltage_threshold": round(vol_thr, 3),
            "harmonic_threshold": round(harm_thr, 3),
            "statistical_threshold": round(stat_thr, 3)
        }
        mlbase.log_metrics(thresholds)
        logger.info(f"Computed thresholds: {thresholds}")

        # 6. Calculate Quality Index
        quality_index = (1.0 / (cur_std + vol_std + 1e-6)) * np.log1p(df.shape[0])
        mlbase.log_metrics({"quality_index": quality_index})
        logger.info(f"📈 Quality index: {quality_index:.4f}")

        # 7. Build and Log YAML Config
        output_yaml = f"winding_fault_configs_m{args.machine_id}.yaml"
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        yaml_path = os.path.join(ARTIFACTS_DIR, output_yaml)
        
        # Ensure values are native Python floats for YAML dump
        yaml_thresholds = {k: float(v) for k, v in thresholds.items()}
        
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_thresholds, f, sort_keys=False, default_flow_style=False)
        
        mlbase.log_artifact(run_id, local_path=yaml_path)
        logger.info(f"📜 Saved and logged config to MLflow: {output_yaml}")
        
        # 8. Set Run Notes
        all_metrics = {**imbalance_metrics, **thresholds, "quality_index": quality_index}
        notes = generate_run_notes(params_to_log, all_metrics, output_yaml)
        mlflow.set_tag("mlflow.note.content", notes)

    except Exception as e:
        logger.error(f"❌ An error occurred during the run: {e}", exc_info=True)
        raise
    finally:
        # 9. End MLflow Run
        mlflow.end_run()
        logger.info("🏁 Run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and log a winding fault detection config to MLflow.")

    # example usage:
    # python winding_model_training.py --tenant-id "28" --machine-id "257" --dataset-filename "iotts.harmonics_257.csv"
    # example production run
    # python winding_model_training.py --tenant-id "28" --machine-id "257" --dataset-filename "iotts.harmonics_257.csv" --run-tag @production
    # exmple testing run
    # python winding_model_training.py --tenant-id "28" --machine-id "257" --dataset-filename "iotts.harmonics_257.csv" --run-tag @testing

    # example usage:
    # python winding_model_training.py --tenant-id 28 --machine-id 257 --dataset-filename ignore.csv --source api --api-base-url "https://iot.zolnoi.app" --api-key "mqdm_CgwNaRsb62Ziy5ePw" --start-datetime-utc "2025-08-01T01:30:00Z" --end-datetime-utc "2025-08-02T01:30:00Z"
    
    # --- Required Arguments ---
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant ID for the machine.")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID for the new equipment.")
    parser.add_argument("--dataset-filename", type=str, required=True, help=f"Filename of the CSV dataset (must be in the '{DATA_DIR}/' folder).")
    
    # --- Optional Model/Knob Arguments ---
    parser.add_argument("--run-name", type=str, default="winding_fault_config_v1", help="Name for the approach/run.")
    
    parser.add_argument("--run-tag", type=str, choices=["@production", "@testing"], default="@testing", help="Tag for the MLflow run: '@production' or '@testing'.")

    # --- Data source toggle ---
    parser.add_argument("--source", choices=["local", "api"], default="local",
                        help="Where to load data from (default: local CSV).")
    parser.add_argument("--api-base-url", type=str, default=os.getenv("IOT_API_BASE_URL"),
                        help="API base URL (e.g., https://iot.zolnoi.app). Defaults to env IOT_API_BASE_URL.")
    parser.add_argument("--api-key", type=str, default=os.getenv("IOT_API_KEY"),
                        help="API key. Defaults to env IOT_API_KEY.")
    parser.add_argument("--start-datetime-utc", type=str, default=None,
                        help="UTC ISO8601 start (e.g., 2025-08-01T01:30:00Z). Required if --source api.")
    parser.add_argument("--end-datetime-utc", type=str, default=None,
                        help="UTC ISO8601 end (e.g., 2025-08-02T01:30:00Z). Required if --source api.")
    parser.add_argument("--max-span-hours", type=int, default=24,
                        help="Max hours per API chunk (≤24).")


    cli_args = parser.parse_args()
    main(cli_args)
