import logging
import os
from typing import Any, Dict, Optional

import joblib
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from tensorflow.keras.models import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def get_model_version_by_alias_safe(client, model_name, alias):
    """Safely get model version by alias, handling not found exceptions."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except mlflow.exceptions.RestException as e:
        if "alias" in str(e) and "not found" in str(e):
            return None
        raise


class MLflowService:
    """
    Unified MLflow service for experiment tracking and model management.
    Combines experiment setup, run tracking, and model operations.
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialize MLflow connection and experiment.
        
        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.run_id: Optional[str] = None
        self.client: Optional[MlflowClient] = None
        
        self._setup_mlflow_connection()
        self._setup_experiment()
    
    def _setup_mlflow_connection(self):
        """Setup MLflow connection with environment variables."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not all([tracking_uri, username, password]):
            raise ValueError(
                "Missing required environment variables. Please check .env.local file."
            )

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def _setup_experiment(self):
        """Setup or get existing experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
        
        run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        self.run_id = run.info.run_id
        logger.info(f"🚀 Started MLflow Run: {self.run_id} in Experiment: '{self.experiment_name}'")
        return run
    
    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the current run."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric to the current run."""
        mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to the current run."""
        if self.run_id:
            self.client.log_artifact(self.run_id, local_path, artifact_path)
        else:
            mlflow.log_artifact(local_path, artifact_path)
    
    def log_keras_model(self, model: Model, name: str = "model", signature=None) -> None:
        """Log a Keras model to MLflow."""
        mlflow.keras.log_model(model, name, signature=signature)
    
    def register_model(self, model_uri: str, name: str) -> str:
        """
        Register a model in the Model Registry.
        
        Args:
            model_uri (str): URI of the model to register
            name (str): Name for the registered model
            
        Returns:
            str: Version of the registered model
        """
        try:
            result = mlflow.register_model(model_uri, name)
            logger.info(f"Successfully registered model version {result.version}")
            return result.version
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise
    
    def promote_best_model(self, model_name: str) -> None:
        """
        Promote the best model (lowest val_loss) to @production alias.
        
        Args:
            model_name (str): Registered model name in MLflow
        """
        # Get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{self.experiment_name}' not found.")
            return

        # Get recent runs
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=50
        )

        # Filter only those with val_loss
        valid_runs = runs_df[runs_df["metrics.val_loss"].notna()]
        if valid_runs.empty:
            logger.warning("No valid runs with 'val_loss' found.")
            return

        # Select best run (lowest val_loss)
        best_run = valid_runs.sort_values("metrics.val_loss", ascending=True).iloc[0]
        best_run_id = best_run.run_id
        best_val_loss = best_run["metrics.val_loss"]

        # Check if already in production
        existing = get_model_version_by_alias_safe(self.client, model_name, "production")
        if existing and existing.run_id == best_run_id:
            logger.info(f"Model from run {best_run_id} is already in @production.")
            return

        # Register and promote
        model_uri = f"runs:/{best_run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        self.client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=result.version
        )

        logger.info(f"Promoted run {best_run_id} to alias @production (val_loss = {best_val_loss:.4f})")
    
    def save_model_locally(self, model, filename: str, directory: str = "models"):
        """
        Save a model to a local directory using joblib.
        
        Args:
            model: The model object to save
            filename (str): The filename to use for saving the model
            directory (str): The directory to save the model in (default: 'models')
        """
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, filename)
        joblib.dump(model, model_path)
        logger.info(f"Model saved locally at {model_path}")
    
    def download_best_model(self, metric: str = "val_loss", local_dir: str = "best_model"):
        """
        Download the best model from the experiment.
        
        Args:
            metric (str): Metric to select the best model (default: 'val_loss')
            local_dir (str): Local directory to download the model
        """
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                logger.warning(f"Experiment '{self.experiment_name}' not found.")
                return
                
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="",
                run_view_type=1,
                max_results=1000,
                order_by=[f"metrics.{metric} ASC" if metric == "val_loss" else f"metrics.{metric} DESC"],
            )
            
            if not runs:
                logger.warning(f"No runs found for experiment '{self.experiment_name}'.")
                return
                
            best_run = runs[0]
            logger.info(f"Best run ID: {best_run.info.run_id}, {metric}: {best_run.data.metrics.get(metric)}")
            
            model_uri = f"runs:/{best_run.info.run_id}/model"
            logger.info(f"Downloading best model from {model_uri} to {local_dir} ...")
            mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=local_dir)
            logger.info(f"Best model downloaded to {local_dir} directory")
            
        except Exception as e:
            logger.error(f"Failed to download best model: {e}")
            raise
