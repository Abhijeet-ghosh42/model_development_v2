import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import yaml

import joblib
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
import sys
import typer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.mlflow_service import MLflowService
from utils.logging import logger

# Configure logging and warnings
warnings.filterwarnings("ignore")

# Create Typer app
app = typer.Typer(help="Run an anomaly detection experiment with LSTM Autoencoder.")

# --- Constants ---
DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"
FEATURES = ['cur1', 'cur2', 'cur3', 'pf1', 'pf2', 'pf3', 'vol1', 'vol2', 'vol3', 'thdi1', 'thdi2', 'thdi3', 'freq']
DEFAULT_SHAP_SAMPLES = 100
DEFAULT_RANDOM_STATE = 42


@dataclass
class TrainingConfig:
    """Configuration class for anomaly detection training."""
    run_name: str
    tenant_id: int
    machine_id: int
    model_name: str
    dataset_filename: str
    train_start_date: Optional[str]
    window_size: int
    stoppage_threshold: float
    threshold_sigma_multiplier: float
    train_end_date: Optional[str]
    latent_dim: int
    epochs: int
    batch_size: int
    register: bool
    run_tag: str
    data_dir: str = DATA_DIR
    artifacts_dir: str = ARTIFACTS_DIR
    features: List[str] = None
    shap_samples: int = DEFAULT_SHAP_SAMPLES
    random_state: int = DEFAULT_RANDOM_STATE
    
    def __post_init__(self):
        if self.features is None:
            self.features = FEATURES


# --- Custom Callbacks ---

class EveryNEpochsModelCheckpoint(Callback):
    """Custom callback to save model only every N epochs."""
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, n_epochs=5, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.best = float('inf') if monitor == 'val_loss' else float('-inf')
        self.best_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                logger.warning(f"Can save best model only with {self.monitor} available, skipping.")
            return
            
        # Check if this is a better model
        is_better = (current < self.best) if self.monitor == 'val_loss' else (current > self.best)
            
        # Save only every N epochs and if it's the best model
        if (epoch + 1) % self.n_epochs == 0:
            if not self.save_best_only or is_better:
                self.model.save(self.filepath)
                if self.verbose > 0:
                    logger.info(f"Epoch {epoch + 1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, saving model to {self.filepath}")
            elif self.verbose > 0:
                logger.info(f"Epoch {epoch + 1}: {self.monitor} did not improve from {self.best:.4f}, not saving model")

        if is_better:
            self.best = current
            self.best_epoch = epoch

# --- Data Processing Classes ---

class DataProcessor:
    """Handles data loading, preprocessing, and sequence creation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scaler: Optional[MinMaxScaler] = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate data from CSV file."""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
                
            logger.info(f"Loading data from {filepath}...")
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['timestamp'] + self.config.features
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.dropna(subset=self.config.features, inplace=True)
            
            if df.empty:
                raise ValueError("No data remaining after removing NaN values")
                
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
            
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            raise
    
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on stoppage threshold."""
        logger.info(f"Filtering data with stoppage current threshold >= {self.config.stoppage_threshold} A...")
        
        current_cols = ['cur1', 'cur2', 'cur3']
        mask = df[current_cols].ge(self.config.stoppage_threshold).all(axis=1)
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            raise ValueError(f"No data remaining after filtering with threshold {self.config.stoppage_threshold}")
            
        logger.info(f"Filtered to {len(filtered_df)} records")
        return filtered_df
    
    def normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """Normalize features using MinMaxScaler."""
        logger.info("Normalizing features with MinMaxScaler...")
        
        self.scaler = MinMaxScaler()
        df_normalized = df.copy()
        df_normalized[self.config.features] = self.scaler.fit_transform(df[self.config.features])
        
        return df_normalized, self.scaler
    
    def create_shap_background(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create SHAP background dataset."""
        return df[self.config.features].sample(
            n=min(self.config.shap_samples, len(df)), 
            random_state=self.config.random_state
        )
    
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Convert time-series data into overlapping sequences."""
        logger.info(f"Creating sequences with window size {self.config.window_size}...")
        
        if len(data) < self.config.window_size:
            raise ValueError(f"Data length ({len(data)}) is less than window size ({self.config.window_size})")
            
        sequences = np.array([
            data[i:i+self.config.window_size] 
            for i in range(len(data) - self.config.window_size + 1)
        ])
        
        logger.info(f"Created {len(sequences)} sequences")
        return sequences
    
    def process_data(self, filepath: str) -> Tuple[pd.DataFrame, MinMaxScaler, pd.DataFrame]:
        """Complete data processing pipeline."""
        df = self.load_data(filepath)
        df = self.filter_data(df)
        df, scaler = self.normalize_data(df)
        shap_background = self.create_shap_background(df)
        
        return df, scaler, shap_background

class ModelBuilder:
    """Handles LSTM Autoencoder model construction and training."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def build_autoencoder(self, timesteps: int, n_features: int) -> Model:
        """Build and compile the LSTM Autoencoder model."""
        logger.info(f"Building LSTM Autoencoder with latent dim {self.config.latent_dim}...")
        
        try:
            input_layer = Input(shape=(timesteps, n_features))
            encoded = LSTM(self.config.latent_dim, activation='relu', return_sequences=False)(input_layer)
            decoded = RepeatVector(timesteps)(encoded)
            decoded = LSTM(self.config.latent_dim, activation='relu', return_sequences=True)(decoded)
            output_layer = TimeDistributed(Dense(n_features))(decoded)
            
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer='adam', loss='mse')
            
            logger.info(f"Model built successfully. Total parameters: {model.count_params():,}")
            return model
            
        except Exception as e:
            logger.exception(f"Error building model: {e}")
            raise
    
    def train_model(self, model: Model, X_train: np.ndarray) -> Model:
        """Train the LSTM Autoencoder with callbacks."""
        logger.info(f"Training model for {self.config.epochs} epochs...")
        
        try:
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                EveryNEpochsModelCheckpoint(
                    filepath=os.path.join(self.config.artifacts_dir, 'best_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    n_epochs=5,  # Save only every 5 epochs
                    verbose=1
                )
            ]
            
            # Train the model
            history = model.fit(
                X_train, X_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=0.2,
                shuffle=False,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            return model, history
            
        except Exception as e:
            logger.exception(f"Error training model: {e}")
            raise
    
    def calculate_threshold(self, model: Model, X_train: np.ndarray) -> float:
        """Calculate anomaly threshold based on training data."""
        logger.info("Calculating anomaly threshold...")
        
        try:
            X_train_pred = model.predict(X_train, verbose=0)
            mse_train = np.mean(np.power(X_train - X_train_pred, 2), axis=(1, 2))
            
            threshold = float(
                np.mean(mse_train) + 
                self.config.threshold_sigma_multiplier * np.std(mse_train)
            )
            
            logger.info(f"Anomaly threshold calculated: {threshold:.6f}")
            return threshold
            
        except Exception as e:
            logger.exception(f"Error calculating threshold: {e}")
            raise
    
    def predict_anomalies(self, model: Model, X_test: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies on test data."""
        logger.info("Predicting anomalies on test set...")
        
        try:
            X_test_pred = model.predict(X_test, verbose=0)
            mse_test = np.mean(np.power(X_test - X_test_pred, 2), axis=(1, 2))
            anomaly_labels = mse_test > threshold
            
            anomaly_count = int(np.sum(anomaly_labels))
            logger.info(f"Found {anomaly_count} anomalies out of {len(X_test)} samples")
            
            return mse_test, anomaly_labels
            
        except Exception as e:
            logger.exception(f"Error predicting anomalies: {e}")
            raise

class ConfigBuilder:
    """Handles configuration file generation with baselines and limits."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_cols = FEATURES
    
    def calculate_baselines(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline values for each signal type."""
        # Filter only numeric columns
        # numeric_columns = df.select_dtypes(include=[np.number]).columns
        # logger.info(f"Numeric columns: {list(numeric_columns)}")
        
        # Calculate mean for numeric columns only
        baselines = df[self.feature_cols].mean().to_dict()
        # Convert numpy types to regular Python types for YAML serialization
        baselines = {k: float(v) for k, v in baselines.items()}
        logger.info(f"Baselines: {baselines}")
        
        return baselines
    
    # def calculate_energy_baselines(self, baselines: Dict[str, float]) -> List[float]:
    #     """Create energy parameter baselines list."""
    #     return ([baselines['cur']] * 3 + 
    #             [baselines['pf']] * 3 + 
    #             [baselines['vol']] * 3 + 
    #             [baselines['thd']] * 3 + 
    #             [baselines['freq']])
    
    def calculate_allowed_extremes(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate allowed extremes (3-sigma limits) for each signal."""
        allowed_extremes = {}
        
        # Process current, power factor, voltage, and THD signals
        for signal in self.feature_cols:
            # combined = pd.concat([df[f"{signal}{i}"] for i in range(1, 4)])
            # mean, std = float(combined.mean()), float(combined.std())
            mean, std = float(df[signal].mean()), float(df[signal].std())
            allowed_extremes[f"{signal}"] = [mean - 3*std, mean + 3*std]
        
        # # Process frequency separately
        # mean_freq, std_freq = float(df['freq'].mean()), float(df['freq'].std())
        # allowed_extremes["freq"] = [
        #     round(mean_freq - 3*std_freq, 2), 
        #     round(mean_freq + 3*std_freq, 2)
        # ]
        
        return allowed_extremes
    
    def build_config(self, df: pd.DataFrame, threshold: float) -> Dict:
        """Build complete configuration dictionary."""
        logger.info("Building configuration file with signal baselines and limits...")
        
        try:
            baselines = self.calculate_baselines(df)
            allowed_extremes = self.calculate_allowed_extremes(df)
            
            # Convert baselines dict to list format for backward compatibility
            # energy_baselines = list(baselines.values())
            
            config = {
                "anomaly_threshold": 
                {
                    "low": threshold,
                    "moderate": threshold + 0.0005,
                    "high": threshold + 0.0015
                },
                "feature_cols": self.feature_cols,
                "model_name": self.config.model_name,
                "window_size": self.config.window_size,
                "stoppage_current_threshold": self.config.stoppage_threshold,
                "energy_parameter_baselines": baselines,
                "allowed_energy_extremes": allowed_extremes,
            }
            
            return config
            
        except Exception as e:
            logger.exception(f"Error building configuration: {e}")
            raise
    
    def save_config(self, config: Dict, config_path: str) -> None:
        """Save configuration to YAML file."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=True)
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.exception(f"Error saving configuration: {e}")
            raise



# --- Main Execution Logic ---

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
        # self.model_name = f"{usecase}/{config.tenant_id}/{config.machine_id}"
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


@app.command()
def run_experiment(
    run_name: str = typer.Option("LSTM Anomaly Run", "--run-name", "-r", help="MLflow run name"),
    tenant_id: int = typer.Option(28, "--tenant-id", "-t", help="Tenant ID"),
    machine_id: int = typer.Option(257, "--machine-id", "-m", help="Machine ID"),
    usecase: str = typer.Option("anomaly_health_monitoring", "--usecase", "-uc", help="Usecase"),
    dataset_filename: str = typer.Option("iotts.energy_257.csv", "--dataset-filename", "-d", help="Dataset filename"),
    train_start_date: Optional[str] = typer.Option(None, "--train-start-date", "-st", help="Training start date (YYYY-MM-DD)"),
    window_size: int = typer.Option(15, "--window-size", "-w", help="Window size for sequences"),
    stoppage_threshold: float = typer.Option(15.0, "--stoppage-threshold", "-sth", help="Stoppage current threshold in Amperes"),
    threshold_sigma_multiplier: float = typer.Option(7.0, "--threshold-sigma-multiplier", "-tsm", help="Sigma multiplier for anomaly threshold"),
    train_end_date: Optional[str] = typer.Option(None, "--train-end-date", "-et", help="Training end date (YYYY-MM-DD)"),
    latent_dim: int = typer.Option(32, "--latent-dim", "-l", help="Latent dimension for LSTM"),
    epochs: int = typer.Option(100, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for training"),
    register: bool = typer.Option(False, "--register", help="Register the model in MLflow Model Registry"),
    run_tag: str = typer.Option("@production", "--run-tag", "-rt", help="Run tag")
):
    """Run an anomaly detection experiment with LSTM Autoencoder."""
    try:
        # Create configuration from CLI parameters
        config = TrainingConfig(
            run_name=run_name,
            tenant_id=tenant_id,
            machine_id=machine_id,
            model_name=f"{usecase}_{tenant_id}_{machine_id}",
            dataset_filename=dataset_filename,
            train_start_date=train_start_date,
            window_size=window_size,
            stoppage_threshold=stoppage_threshold,
            threshold_sigma_multiplier=threshold_sigma_multiplier,
            train_end_date=train_end_date,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            register=register,
            run_tag=run_tag
        )
        
        # Run the pipeline
        run_anomaly_detection_pipeline(config)
        
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()