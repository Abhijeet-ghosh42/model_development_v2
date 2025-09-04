import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
import logging
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow_base import MLflowBase
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
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

FEATURES = ['cur1', 'cur2', 'cur3', 'pf1', 'pf2', 'pf3', 'vol1', 'vol2', 'vol3', 'thdi1', 'thdi2', 'thdi3', 'freq']

# --- Helper Functions ---

def load_and_prepare_data(filepath, stoppage_threshold):
    """Loads, filters, and normalizes the raw energy data."""
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.dropna(subset=FEATURES, inplace=True)
    
    logger.info(f"Filtering data with stoppage current threshold >= {stoppage_threshold} A...")
    df = df[(df['cur1'] >= stoppage_threshold) & (df['cur2'] >= stoppage_threshold) & (df['cur3'] >= stoppage_threshold)]
    
    logger.info("Normalizing features with MinMaxScaler...")
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    
    shap_background = df[FEATURES].sample(n=100, random_state=42)
    
    return df, scaler, shap_background

def create_sequences(data, window_size):
    """Converts a time-series dataframe into overlapping sequences."""
    logger.info(f"Creating sequences with window size {window_size}...")
    return np.array([data[i:i+window_size] for i in range(len(data) - window_size + 1)])

def build_autoencoder(timesteps, n_features, latent_dim):
    """Builds and compiles the LSTM Autoencoder model."""
    logger.info(f"Building LSTM Autoencoder with latent dim {latent_dim}...")
    input_layer = Input(shape=(timesteps, n_features))
    encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(input_layer)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(decoded)
    output_layer = TimeDistributed(Dense(n_features))(decoded)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_and_save_config(config_path, raw_filepath, stoppage_threshold, threshold):
    """Calculates baselines and signal limits to build the final config JSON."""
    logger.info("Building configuration file with signal baselines and limits...")
    df_raw = pd.read_csv(raw_filepath)
    df_raw = df_raw[(df_raw['cur1'] >= stoppage_threshold) & (df_raw['cur2'] >= stoppage_threshold) & (df_raw['cur3'] >= stoppage_threshold)]

    baselines = {
        'cur': round(df_raw[['cur1', 'cur2', 'cur3']].mean().mean(), 2),
        'pf': round(df_raw[['pf1', 'pf2', 'pf3']].mean().mean(), 2),
        'vol': round(df_raw[['vol1', 'vol2', 'vol3']].mean().mean(), 2),
        'thd': round(df_raw[['thdi1', 'thdi2', 'thdi3']].mean().mean(), 2),
        'freq': round(df_raw['freq'].mean(), 2),
    }
    
    energy_baselines = [baselines['cur']]*3 + [baselines['pf']]*3 + [baselines['vol']]*3 + [baselines['thd']]*3 + [baselines['freq']]

    allowed_extremes = {}
    for signal in ['cur', 'pf', 'vol', 'thdi']:
        combined = pd.concat([df_raw[f"{signal}{i}"] for i in range(1, 4)])
        mean, std = combined.mean(), combined.std()
        for i in range(1, 4):
            allowed_extremes[f"{signal}{i}"] = [round(mean - 3*std, 2), round(mean + 3*std, 2)]
    
    mean_freq, std_freq = df_raw['freq'].mean(), df_raw['freq'].std()
    allowed_extremes["freq"] = [round(mean_freq - 3*std_freq, 2), round(mean_freq + 3*std_freq, 2)]

    config = {
        "anomaly_threshold": threshold,
        "anomaly_moderate_severity_threshold": threshold + 0.0005,
        "anomaly_high_severity_threshold": threshold + 0.0015,
        "stoppage_current_threshold": stoppage_threshold,
        "energy_parameter_baselines": energy_baselines,
        "Allowed_energy_extremes": allowed_extremes,
        "anomaly_model_path": "model",
        "scaler_path": "scaler.pkl",
        "shap_dataset_path": "shap_background.csv"
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")


# --- Main Execution Logic ---

def main(args):
    """Main function to run the anomaly detection model training pipeline."""
    if mlflow.active_run():
        mlflow.end_run()
        
    # 1. MLflow Setup
    usecase = "anomaly_health_monitoring"
    experiment_name = f"{usecase}/{args.tenant_id}/{args.machine_id}"
    model_name = f"{args.tenant_id}_{args.machine_id}_{usecase}"
    mlflow_wrapper = MLflowBase(experiment_name=experiment_name)
    
    with mlflow_wrapper.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_tag", args.run_tag)
        logger.info(f"🚀 Started MLflow Run: {run_id} in Experiment: '{experiment_name}'")
        
        # # 2. Data Preparation
        # dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        # df, scaler, shap_background = load_and_prepare_data(dataset_path, args.stoppage_threshold)

        # # check if training duration range is provided, if not I will set it to range of dataset
        # # Dynamically set training date range if not provided 
        # train_start_date = args.train_start_date
        # train_end_date = args.train_end_date

        # if train_start_date is None:
        #     train_start_date = df['timestamp'].min().strftime('%Y-%m-%d')
        #     logger.info(f"No start date provided. Defaulting to dataset start date: {train_start_date}")

        # if train_end_date is None:
        #     train_end_date = df['timestamp'].max().strftime('%Y-%m-%d')
        #     logger.info(f"No end date provided. Defaulting to dataset end date: {train_end_date}")
        

        # df_train = df[(df["timestamp"] >= args.train_start_date) & (df["timestamp"] <= args.train_end_date)].copy()
        # X_train = create_sequences(df_train[FEATURES].values, args.window_size)
        # X_test = create_sequences(df[FEATURES].values, args.window_size)

        # # 3. Log Parameters
        # mlflow_wrapper.log_params({
        #     "dataset_filename": args.dataset_filename, "window_size": args.window_size,
        #     "latent_dim": args.latent_dim, "epochs": args.epochs, "batch_size": args.batch_size,
        #     # "train_start_date": args.train_start_date, "train_end_date": args.train_end_date,
        #     "train_start_date": train_start_date, "train_end_date": train_end_date,
        #     "threshold_sigma_multiplier": args.threshold_sigma_multiplier
        # })

        # 2. Data Preparation
        dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        df, scaler, shap_background = load_and_prepare_data(dataset_path, args.stoppage_threshold)

        # Dynamically set training date range if not provided
        train_start_date = args.train_start_date
        train_end_date = args.train_end_date

        if train_start_date is None:
            train_start_date = df['timestamp'].min().strftime('%Y-%m-%d')
            logger.info(f"No start date provided. Defaulting to dataset start date: {train_start_date}")

        if train_end_date is None:
            train_end_date = df['timestamp'].max().strftime('%Y-%m-%d')
            logger.info(f"No end date provided. Defaulting to dataset end date: {train_end_date}")

        df_train = df[(df["timestamp"] >= train_start_date) & (df["timestamp"] <= train_end_date)].copy()
        X_train = create_sequences(df_train[FEATURES].values, args.window_size)
        X_test = create_sequences(df[FEATURES].values, args.window_size)
        
        if X_train.shape[0] == 0:
            error_msg = f"Training data is empty after filtering for the date range '{train_start_date}' to '{train_end_date}'. Please check the dates."
            raise ValueError(error_msg)

        # 3. Log Parameters
        mlflow_wrapper.log_params({
            "dataset_filename": args.dataset_filename, "window_size": args.window_size,
            "latent_dim": args.latent_dim, "epochs": args.epochs, "batch_size": args.batch_size,
            "train_start_date": train_start_date, 
            "train_end_date": train_end_date,
            "threshold_sigma_multiplier": args.threshold_sigma_multiplier
        })
        
        # 4. Model Training
        model = build_autoencoder(X_train.shape[1], X_train.shape[2], args.latent_dim)
        history = model.fit(
            X_train, X_train, epochs=args.epochs, batch_size=args.batch_size,
            validation_split=0.2, shuffle=False, verbose=1
        )
        
        for epoch, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)

        # 5. Threshold Calculation and Metrics
        logger.info("Predicting on train set to calculate anomaly threshold...")
        X_train_pred = model.predict(X_train)
        mse_train = np.mean(np.power(X_train - X_train_pred, 2), axis=(1, 2))
        threshold = float(np.mean(mse_train) + args.threshold_sigma_multiplier * np.std(mse_train))

        logger.info("Predicting on test set to count anomalies...")
        X_test_pred = model.predict(X_test)
        mse_test = np.mean(np.power(X_test - X_test_pred, 2), axis=(1, 2))
        anomaly_labels = mse_test > threshold
        
        mlflow_wrapper.log_metrics({
            "threshold": threshold, "mean_train_mse": float(np.mean(mse_train)),
            "std_train_mse": float(np.std(mse_train)), "anomaly_count": int(np.sum(anomaly_labels))
        })
        logger.info(f"Anomaly threshold set to {threshold:.6f}. Found {np.sum(anomaly_labels)} anomalies.")
        
        # 6. Save and Log Artifacts
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        model_path = os.path.join(ARTIFACTS_DIR, "anomaly_model.h5")
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        shap_path = os.path.join(ARTIFACTS_DIR, "shap_background.csv")
        config_path = os.path.join(ARTIFACTS_DIR, "config.json")
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        shap_background.to_csv(shap_path, index=False)
        
        build_and_save_config(config_path, dataset_path, args.stoppage_threshold, threshold)

        mlflow.keras.log_model(model, artifact_path="model")
        mlflow_wrapper.log_artifact(run_id, scaler_path)
        mlflow_wrapper.log_artifact(run_id, shap_path)
        mlflow_wrapper.log_artifact(run_id, config_path)
        logger.info("All artifacts saved and logged to MLflow.")

        # 7. Register and Promote Model
        model_uri = f"runs:/{run_id}/model"
        mlflow_wrapper.register_model(model_uri, model_name)
        logger.info(f"Model registered as '{model_name}'.")
        
        mlflow_wrapper.promote_best_model(model_name=model_name)
        logger.info(f"Promotion check complete for model '{model_name}'.")
        
        logger.info("🏁 Run finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM Autoencoder for anomaly detection and log to MLflow.")
    
    # example usage:
    # python anomaly_health_model_training.py --tenant-id "27" --machine-id "242" --dataset-filename "iotts.energy_242.csv" --epochs 50 --window-size 15 --latent-dim 32
    # example production run:
    # python anomaly_health_model_training.py ... --run-tag @production
    # example testing run:
    # python anomaly_health_model_training.py ... --run-tag @testing

    # Required Arguments
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant ID.")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID.")
    parser.add_argument("--dataset-filename", type=str, required=True, help=f"Dataset CSV filename in '{DATA_DIR}/'.")
    
    # Optional Training Arguments
    parser.add_argument("--run-name", type=str, default="lstm_autoencoder_v1", help="Name for the MLflow run.")
    parser.add_argument("--stoppage-threshold", type=float, default=15.0, help="Current threshold to filter out machine stoppage periods.")
    parser.add_argument("--window-size", type=int, default=15, help="Number of timesteps in each sequence.")
    parser.add_argument("--latent-dim", type=int, default=32, help="Dimension of the LSTM autoencoder's bottleneck.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    # parser.add_argument("--train-start-date", type=str, default="2025-05-15", help="Start date for the training data slice (YYYY-MM-DD).")
    # parser.add_argument("--train-end-date", type=str, default="2025-05-28", help="End date for the training data slice (YYYY-MM-DD).")
    parser.add_argument("--train-start-date", type=str, default=None, help="Start date for training (YYYY-MM-DD). Defaults to the dataset's start date if not provided.")
    parser.add_argument("--train-end-date", type=str, default=None, help="End date for training (YYYY-MM-DD). Defaults to the dataset's end date if not provided.")
    parser.add_argument("--threshold-sigma-multiplier", type=int, default=7, help="Multiplier for standard deviation to set the anomaly threshold.")

    parser.add_argument("--run-tag", type=str, choices=["@production", "@testing"], default="@testing", help="Tag for the MLflow run: '@production' or '@testing'.")

    cli_args = parser.parse_args()
    main(cli_args)