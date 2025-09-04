# process_parametric_alerts_model_training.py
# -----------------------------------------------------------------------------
# Generate and log a process parametric alert configuration for a given machine.
# Helps onboard new extruder machines dynamically.
#
# Load dataset -> derive thresholds
# Auto-detect barrel/die zones
# Build YAML config (machine-specific)
# Save + log config + metrics in MLflow
# -----------------------------------------------------------------------------

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import logging
import warnings
from typing import Dict, Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow_base import MLflowBase

# --- Pre-Script Setup ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# DATA_DIR = "data"
# ARTIFACTS_DIR = "artifacts"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")


# --- Helper Functions --------------------------------------------------------

def derive_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """
    Derive machine-specific default thresholds from dataset statistics.
    For now: basic stdev-based rules. Can be extended.
    """
    thr = {}

    # Example: screw rpm deviation tolerance = 3 * std
    for col in ["ext_actrpm", "fdr_actrpm"]:
        if col in df.columns:
            thr[f"{col}_tolerance"] = float(df[col].std() * 3)

    # Melt pressure variation
    if "meltpres" in df.columns:
        thr["meltpres_tolerance"] = float(df["meltpres"].std() * 3)

    # Melt temperature variation
    if "melttemp" in df.columns:
        thr["melttemp_tolerance"] = float(df["melttemp"].std() * 3)

    # Default fallbacks
    thr.setdefault("setpoint_tolerance", 5)
    thr.setdefault("stability_tolerance", 4)
    thr.setdefault("spike_tolerance", 6)
    thr.setdefault("z_score_threshold", 1)

    return thr


# --- Main Execution Logic ----------------------------------------------------

def main(args):
    if mlflow.active_run():
        mlflow.end_run()

    # 1. Setup MLflow
    experiment_name = f"process_parametric_alert_monitoring/{args.tenant_id}/{args.machine_id}"
    mlbase = MLflowBase(experiment_name)
    run = mlbase.start_run(run_name=args.run_name)
    run_id = run.info.run_id
    mlflow.set_tag("run_tag", args.run_tag)
    logger.info(f"🚀 Started MLflow Run {run_id} in Experiment '{experiment_name}'")

    try:
        # 2. Load Dataset
        dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        logger.info(f"💾 Loading data: {dataset_path}")
        df = pd.read_csv(dataset_path)

        # 3. Derive Thresholds
        thresholds = derive_thresholds(df)

        # 4. Log Parameters (tenant/machine info + thresholds)
        mlbase.log_params({
            "tenant_id": args.tenant_id,
            "machine_id": args.machine_id,
            "dataset_filename": args.dataset_filename,
            "run_name": args.run_name,
            **{k: round(v, 4) for k, v in thresholds.items()}
        })

        # 5. Build Config
        config = {
            "tenant_id": args.tenant_id,
            "machine_id": args.machine_id,
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
                "hauloff_mpm": "fhoff_actmpm",
            },
            "zones": {
                "bz": [c.replace("_actpv", "") for c in df.columns if isinstance(c, str) and c.startswith("bz") and c.endswith("_actpv")],
                "dz": [c.replace("_actpv", "") for c in df.columns if isinstance(c, str) and c.startswith("dz") and c.endswith("_actpv")],
            },
            "thresholds": thresholds,
        }

        # 6. Save Config to YAML
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        yaml_filename = f"process_alerts_config_{args.machine_id}.yaml"
        yaml_path = os.path.join(ARTIFACTS_DIR, yaml_filename)
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        mlbase.log_artifact(run_id, local_path=yaml_path)
        logger.info(f"📜 Saved and logged config to MLflow: {yaml_filename}")

        # 7. Set minimal run notes (just artifact path)
        mlflow.set_tag("mlflow.note.content", f"Config artifact: {yaml_filename}")

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise
    finally:
        mlflow.end_run()
        logger.info("🏁 Run finished.")


# --- CLI Entrypoint ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate process parametric alerts config and log to MLflow.")

    # Example Usage:
    # python process_parametric_alerts_model_training.py --tenant-id "21" --machine-id "166" --dataset-filename "iotts.process.csv" --run-tag @production

    # Required
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant ID")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID")
    parser.add_argument("--dataset-filename", type=str, required=True, help=f"CSV file in {DATA_DIR}/")

    # Optional
    parser.add_argument("--run-name", type=str, default="process_alerts_v1", help="MLflow run name")
    parser.add_argument("--run-tag", type=str, choices=["@production", "@testing"], default="@testing", help="Run tag")

    cli_args = parser.parse_args()
    main(cli_args)
