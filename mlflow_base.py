import logging
import os
from typing import Any, Dict, Optional

import joblib
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_model_version_by_alias_safe(client, model_name, alias):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except mlflow.exceptions.RestException as e:
        if "alias" in str(e) and "not found" in str(e):
            return None
        raise  # raise if it’s another error

class MLflowBase:
    def __init__(self, experiment_name: str):
        """
        Initialize MLflow connection and experiment.

        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        # Load environment variables
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.username = os.getenv("MLFLOW_TRACKING_USERNAME")
        self.password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not all([self.tracking_uri, self.username, self.password]):
            raise ValueError(
                "Missing required environment variables. Please check .env.local file."
            )

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Set experiment
        self.experiment_name = experiment_name
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id

        # Initialize client
        self.client = MlflowClient()

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        return mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the current run."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self, run_id: str, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log an artifact to the current run."""
        self.client.log_artifact(run_id, local_path, artifact_path)

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
            print(str(result))
            return result.version
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise

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
        print(f"Model saved locally at {model_path}")

    def log_model(self, model, artifact_path: str = "model", **kwargs) -> None:
        """
        Log a model in MLflow format (for sklearn models).

        Args:
            model: The model object to log
            artifact_path (str): The artifact path in MLflow (default: 'model')
            **kwargs: Additional keyword arguments for mlflow.sklearn.log_model
        """
        mlflow.sklearn.log_model(model, artifact_path, **kwargs)

    def select_and_download_best_model(
        self, experiment_name: str, metric: str = "accuracy", local_dir: str = "best_model"
    ):
        """
        Select the best model from the given experiment by the specified metric and download its artifact to a local folder.

        Args:
            experiment_name (str): Name of the MLflow experiment
            metric (str): Metric to select the best model (default: 'accuracy')
            local_dir (str): Local directory to download the best model artifact
        """
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"Experiment '{experiment_name}' not found.")
                return
            experiment_id = experiment.experiment_id
            print(f"Searching runs in experiment '{experiment_name}' (ID: {experiment_id})...")
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                run_view_type=1,
                max_results=1000,
                order_by=[f"metrics.{metric} DESC"],
            )
            if not runs:
                print(f"No runs found for experiment '{experiment_name}'.")
                return
            best_run = runs[0]
            print(
                f"Best run ID: {best_run.info.run_id}, {metric}: {best_run.data.metrics.get(metric)}"
            )
            try:
                model_versions = [
                    str(model_version.version)
                    for model_version in mlflow.search_model_versions(
                        filter_string=f"run_id='{best_run.info.run_id}'"
                    )
                ]
                print(
                    "Model versions for best run:",
                    ", ".join(model_versions) if model_versions else "None",
                )
            except Exception as e:
                print(f"Could not retrieve best model version: {e}")
            model_uri = f"runs:/{best_run.info.run_id}/model"
            print(f"Downloading best model from {model_uri} to {local_dir} ...")
            mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=local_dir)
            print(f"Best model downloaded to {local_dir} directory")
        except Exception as e:
            print(f"Failed to select or download best model: {e}")
            import traceback

            traceback.print_exc()

    def promote_best_model(self, model_name: str = "anomaly_lstm_autoencoder") -> None:
        """
        Promote the best model (lowest val_loss) to @production alias.

        Args:
            model_name (str): Registered model name in MLflow (default: 'anomaly_lstm_autoencoder')
        """
        # Step 1: Get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            print(f"Experiment '{self.experiment_name}' not found.")
            return

        # Step 2: Get recent runs
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=50
        )

        # Step 3: Filter only those with val_loss
        valid_runs = runs_df[runs_df["metrics.val_loss"].notna()]
        if valid_runs.empty:
            print("No valid runs with 'val_loss' found.")
            return

        # Step 4: Select best run (lowest val_loss)
        best_run = valid_runs.sort_values("metrics.val_loss", ascending=True).iloc[0]
        best_run_id = best_run.run_id
        best_val_loss = best_run["metrics.val_loss"]

        # Step 5: Check if already in production
        existing = get_model_version_by_alias_safe(self.client, model_name, "production")
        if existing and existing.run_id == best_run_id:
            print(f"Model from run {best_run_id} is already in @production.")
            return

        # Step 6: Register and promote
        # model_uri = f"runs:/{best_run_id}/{model_name}"
        model_uri = f"runs:/{best_run_id}/model"

        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        self.client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=result.version
        )

        print(f" Promoted run {best_run_id} to alias @production (val_loss = {best_val_loss:.4f})")
