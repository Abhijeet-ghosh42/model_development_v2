"""Main pipeline for anomaly detection in machine health monitoring."""

import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model

from machine_health.config import TrainingConfig, ConfigBuilder
from machine_health.data import DataProcessor
from machine_health.models import ModelBuilder
from utils.mlflow_service import MLflowService
from utils.logging import logger


class AnomalyDetectionPipeline:
    """Main pipeline class that orchestrates the entire anomaly detection process."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.model_builder = ModelBuilder(config)
        self.config_builder = ConfigBuilder(config)
        
        # Setup MLflow service
        # Change format from usecase_tenant_id_machine_id to usecase/tenant_id/machine_id
        experiment_name = "/".join(self.config.model_name.rsplit("_", 2))
        self.mlflow_service = MLflowService(experiment_name=experiment_name)
    
    def run(self):
        """Execute the complete anomaly detection pipeline."""
        try:
            with self.mlflow_service.start_run(run_name=self.config.run_name):
                # Set run tag
                self.mlflow_service.set_tag("run_tag", self.config.run_tag)
                
                # 1. Data Preparation
                dataset_path = os.path.join(self.config.data_dir, self.config.dataset_filename)
                df, scaler, shap_background = self.data_processor.process_data(dataset_path)
                
                # Set training date range
                train_start_date, train_end_date = self._set_training_dates(df)
                
                # Filter training data
                df_train = df[(df["timestamp"] >= train_start_date) & (df["timestamp"] <= train_end_date)].copy()
                
                if df_train.empty:
                    raise ValueError(f"Training data is empty after filtering for the date range '{train_start_date}' to '{train_end_date}'")
                
                # Create sequences
                X_train = self.data_processor.create_sequences(df_train[self.config.features].values)
                X_test = self.data_processor.create_sequences(df[self.config.features].values)
                
                # 2. Log Parameters
                self._log_anomaly_parameters({
                    "train_start_date": train_start_date,
                    "train_end_date": train_end_date
                })
                
                # 3. Model Training
                model = self.model_builder.build_autoencoder(X_train.shape[1], X_train.shape[2])
                model, history = self.model_builder.train_model(model, X_train)
                
                # Log training metrics
                self._log_training_metrics(history)
                
                # 4. Threshold Calculation and Anomaly Detection
                threshold = self.model_builder.calculate_threshold(model, X_train)
                mse_test, anomaly_labels = self.model_builder.predict_anomalies(model, X_test, threshold)
                
                # Log anomaly metrics
                self._log_anomaly_metrics(
                    threshold=threshold,
                    mean_train_mse=float(np.mean(mse_test)),
                    std_train_mse=float(np.std(mse_test)),
                    anomaly_count=int(np.sum(anomaly_labels))
                )
                
                # 5. Save Artifacts
                self._save_artifacts(model, scaler, shap_background, threshold, dataset_path)
                
                # 6. Log to MLflow
                scaler_path = os.path.join(self.config.artifacts_dir, "scaler.pkl")
                shap_path = os.path.join(self.config.artifacts_dir, "shap_background.csv")
                config_path = os.path.join(self.config.artifacts_dir, "config.yaml")
                
                self._log_model_and_artifacts(model, scaler_path, shap_path, config_path)
                
                # 7. Register and Promote Model
                if self.config.register:
                    self._register_and_promote_model()
                
                logger.info("🏁 Pipeline completed successfully.")
                
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise
    
    def _set_training_dates(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Set training date range based on config or data."""
        train_start_date = self.config.train_start_date
        train_end_date = self.config.train_end_date
        
        if train_start_date is None:
            train_start_date = df['timestamp'].min().strftime('%Y-%m-%d')
            logger.info(f"No start date provided. Defaulting to dataset start date: {train_start_date}")
        
        if train_end_date is None:
            train_end_date = df['timestamp'].max().strftime('%Y-%m-%d')
            logger.info(f"No end date provided. Defaulting to dataset end date: {train_end_date}")
        
        return train_start_date, train_end_date
    
    def _save_artifacts(self, model: Model, scaler: MinMaxScaler, 
                       shap_background: pd.DataFrame, threshold: float, 
                       dataset_path: str):
        """Save all artifacts to disk."""
        try:
            os.makedirs(self.config.artifacts_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(self.config.artifacts_dir, "anomaly_model.h5")
            model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.config.artifacts_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            
            # Save SHAP background
            shap_path = os.path.join(self.config.artifacts_dir, "shap_background.csv")
            shap_background.to_csv(shap_path, index=False)
            
            # Build and save config
            config_path = os.path.join(self.config.artifacts_dir, "config.yaml")
            df_raw = pd.read_csv(dataset_path)
            df_raw = df_raw[(df_raw['cur1'] >= self.config.stoppage_threshold) & 
                           (df_raw['cur2'] >= self.config.stoppage_threshold) & 
                           (df_raw['cur3'] >= self.config.stoppage_threshold)]
            
            config = self.config_builder.build_config(df_raw, threshold)
            self.config_builder.save_config(config, config_path)
            
        except Exception as e:
            logger.exception(f"Error saving artifacts: {e}")
            raise
    
    def _log_anomaly_parameters(self, additional_params: Optional[Dict] = None):
        """Log anomaly detection specific parameters to MLflow."""
        params = {
            "dataset_filename": self.config.dataset_filename,
            "window_size": self.config.window_size,
            "latent_dim": self.config.latent_dim,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "train_start_date": self.config.train_start_date,
            "train_end_date": self.config.train_end_date,
            "threshold_sigma_multiplier": self.config.threshold_sigma_multiplier,
            "stoppage_threshold": self.config.stoppage_threshold
        }
        
        if additional_params:
            params.update(additional_params)
            
        self.mlflow_service.log_params(params)
    
    def _log_training_metrics(self, history):
        """Log training metrics from model history."""
        for epoch, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            self.mlflow_service.log_metric("train_loss", float(train_loss), step=epoch)
            self.mlflow_service.log_metric("val_loss", float(val_loss), step=epoch)
    
    def _log_anomaly_metrics(self, threshold: float, mean_train_mse: float, 
                           std_train_mse: float, anomaly_count: int):
        """Log anomaly detection metrics."""
        self.mlflow_service.log_metrics({
            "threshold": threshold,
            "mean_train_mse": mean_train_mse,
            "std_train_mse": std_train_mse,
            "anomaly_count": anomaly_count
        })
    
    def _log_model_and_artifacts(self, model: Model, scaler_path: str, 
                               shap_path: str, config_path: str):
        """Log model and artifacts to MLflow."""
        try:
            # Log model
            signature = infer_signature(
                np.random.random((1, self.config.window_size, len(self.config.features))),
                np.random.random((1, self.config.window_size, len(self.config.features)))
            )
            self.mlflow_service.log_keras_model(model, name=self.config.model_name, signature=signature)
            
            # Log artifacts
            self.mlflow_service.log_artifact(scaler_path)
            self.mlflow_service.log_artifact(shap_path)
            self.mlflow_service.log_artifact(config_path)
            
            logger.info("All artifacts saved and logged to MLflow.")
            
        except Exception as e:
            logger.exception(f"Error logging model and artifacts: {e}")
            raise
    
    def _register_and_promote_model(self):
        """Register model in MLflow Model Registry and promote if needed."""
        try:
            model_uri = f"runs:/{self.mlflow_service.run_id}/model"
            self.mlflow_service.register_model(model_uri, self.config.model_name)
            logger.info(f"Model registered as '{self.config.model_name}'.")
            
            self.mlflow_service.promote_best_model(model_name=self.config.model_name)
            logger.info(f"Promotion check complete for model '{self.config.model_name}'.")
            
        except Exception as e:
            logger.exception(f"Error registering/promoting model: {e}")
            raise


def run_anomaly_detection_pipeline(config: TrainingConfig) -> None:
    """Run the anomaly detection pipeline with the given configuration."""
    try:
        pipeline = AnomalyDetectionPipeline(config)
        pipeline.run()
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        raise
