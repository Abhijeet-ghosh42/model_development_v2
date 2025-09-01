import argparse
import logging
import os
import time
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow_base import MLflowBase
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# --- Pre-Script Configuration & Constants ---

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"

# --- Model Training Class ---

class AdvancedRotorFaultTrainer:
    """A class to encapsulate the rotor fault model training pipeline."""
    def __init__(self, n_components, svm_nu, svm_gamma):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.svm = OneClassSVM(nu=svm_nu, gamma=svm_gamma)
        self.imputer = SimpleImputer(strategy='mean')
    
    def extract_sideband_features(self, harmonics_data):
        """Extracts sideband harmonic features from the dataframe."""
        harmonics_indices = range(1, 31)
        feature_cols = []

        for phase_type in ['ch', 'vh']:
            for h_index in harmonics_indices:
                for phase_num in range(1, 4):
                    col_name = f"{phase_type}{phase_num}_{h_index}"
                    if col_name in harmonics_data.columns:
                        feature_cols.append(harmonics_data[col_name].values)
                    else:
                        logger.warning(f"Column {col_name} missing, filling with NaN.")
                        feature_cols.append(np.full(len(harmonics_data), np.nan))
        
        return np.column_stack(feature_cols)

    def fit(self, harmonics_data):
        """Fits the entire preprocessing and model pipeline."""
        t0 = time.time()
        logger.info("Extracting sideband features...")
        sideband_data = self.extract_sideband_features(harmonics_data)
        
        logger.info("Imputing missing values...")
        imputed_data = self.imputer.fit_transform(sideband_data)
        
        logger.info("Scaling data...")
        scaled_data = self.scaler.fit_transform(imputed_data)
        
        logger.info("Applying PCA...")
        pca_data = self.pca.fit_transform(scaled_data)
        
        logger.info("Fitting One-Class SVM...")
        self.svm.fit(pca_data)
        
        logger.info(f"✅ Total training time: {time.time() - t0:.2f}s")
        return pca_data
        
# --- Helper Functions ---

def generate_run_notes(params, metrics, model_path, yaml_path):
    """Generates a markdown note for the MLflow run."""
    def _fmt(x, nd=4):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return "NaN"
        try: return f"{float(x):.{nd}f}"
        except: return str(x)
        
    return f"""
# 🌀 Rotor Fault Detection — Run Summary

### Parameters
- **Algorithm**: {params['run_name']}
- **Tenant / Machine**: {params['tenant_id']} / {params['machine_id']}
- **Dataset**: {params['dataset_filename']}
- **PCA components**: {params['pca_n_components']}
- **SVM parameters**: ν = {params['svm_nu']}, γ = {params['svm_gamma']}

### Key Metrics
- **Explained variance (PCA)**: {_fmt(metrics['explained_variance'])}
- **SVM score mean**: {_fmt(metrics['score_mean'])}
- **SVM score std**: {_fmt(metrics['score_std'])}

### Artifacts
- **Model file**: `{model_path}`
- **Config YAML**: `{yaml_path}`
"""

# --- Main Execution Logic ---

def main(args):
    """
    Main function to run the rotor fault model training and MLflow logging pipeline.
    """
    if mlflow.active_run():
        mlflow.end_run()

    # 1. MLflow Setup
    experiment_name = f"rotor_fault_monitoring/{args.tenant_id}/{args.machine_id}"
    mlbase = MLflowBase(experiment_name)
    run = mlbase.start_run(run_name=args.run_name)
    run_id = run.info.run_id
    logger.info(f"🚀 Started MLflow Run: {run_id} in Experiment: '{experiment_name}'")

    try:
        # 2. Log Initial Parameters
        params_to_log = {
            "run_name": args.run_name,
            "dataset_filename": args.dataset_filename,
            "tenant_id": args.tenant_id,
            "machine_id": args.machine_id,
            "pca_n_components": args.n_components,
            "svm_nu": args.svm_nu,
            "svm_gamma": args.svm_gamma,
        }
        mlbase.log_params(params_to_log)

        # 3. Load & Prepare Data
        dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        logger.info(f"💾 Loading data from: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
        
        df = pd.read_csv(dataset_path)
        drop_cols = [c for c in ["_id", "tenant_id", "machine_id", "type", "timestamp"] if c in df.columns]
        harmonics_data = df.drop(columns=drop_cols, errors="ignore")

        # 4. Train Model
        trainer = AdvancedRotorFaultTrainer(args.n_components, args.svm_nu, args.svm_gamma)
        
        data_to_train = harmonics_data
        if args.sample_size:
            logger.warning(f"Using a random sample of {args.sample_size} rows for training.")
            data_to_train = harmonics_data.sample(n=args.sample_size, random_state=42)

        pca_data = trainer.fit(data_to_train)
        
        # 5. Calculate Metrics
        scores = trainer.svm.decision_function(pca_data)
        metrics_to_log = {
            "explained_variance": float(trainer.pca.explained_variance_ratio_.sum()),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }
        mlbase.log_metrics(metrics_to_log)
        logger.info(f"📊 Metrics calculated: Explained Variance = {metrics_to_log['explained_variance']:.4f}")

        # 6. Save Artifacts
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        model_filename = f"rotor_fault_detector_{args.machine_id}.pkl"
        model_path = os.path.join(ARTIFACTS_DIR, model_filename)
        joblib.dump((trainer.imputer, trainer.scaler, trainer.pca, trainer.svm), model_path)
        
        config = {
            "tenant_id": args.tenant_id,
            "machine_id": args.machine_id,
            "pca_n_components": args.n_components,
            "svm_nu": args.svm_nu,
            "svm_gamma": args.svm_gamma,
            "decision_threshold": float(np.percentile(scores, 5)),
            "model_path": model_filename,
        }
        yaml_filename = f"rotor_fault_config_{args.machine_id}.yaml"
        yaml_path = os.path.join(ARTIFACTS_DIR, yaml_filename)
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        # 7. Log Artifacts to MLflow
        mlbase.log_artifact(run_id, local_path=model_path)
        mlbase.log_artifact(run_id, local_path=yaml_path)
        logger.info(f"📜 Saved and logged artifacts to MLflow: {model_filename}, {yaml_filename}")
        
        # 8. Set Run Notes
        notes = generate_run_notes(params_to_log, metrics_to_log, model_filename, yaml_filename)
        mlflow.set_tag("mlflow.note.content", notes)

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}", exc_info=True)
        raise
    finally:
        mlflow.end_run()
        logger.info("🏁 Run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Rotor Fault Detection model and log to MLflow.")

    # Required Arguments
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant ID.")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID.")
    parser.add_argument("--dataset-filename", type=str, required=True, help=f"Dataset CSV filename in '{DATA_DIR}/'.")
    
    # Optional Model/Knob Arguments
    parser.add_argument("--run-name", type=str, default="pca_ocsvm_sidebands_v1", help="Name for the MLflow run.")
    parser.add_argument("--n-components", type=int, default=5, help="Number of PCA components.")
    parser.add_argument("--svm-nu", type=float, default=0.05, help="Nu parameter for OneClassSVM.")
    parser.add_argument("--svm-gamma", type=str, default="scale", help="Gamma parameter for OneClassSVM.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional: use a random sample of N rows for quick training.")

    cli_args = parser.parse_args()
    main(cli_args)
