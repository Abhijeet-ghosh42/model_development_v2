"""Main entry point for machine health monitoring pipeline."""

import sys
from pathlib import Path

# Add the experiments directory to Python path
experiments_dir = Path(__file__).parent.parent
if str(experiments_dir) not in sys.path:
    sys.path.insert(0, str(experiments_dir))

import typer
from typing import Optional

from machine_health import TrainingConfig, run_anomaly_detection_pipeline
from utils.logging import logger

# Create Typer app
app = typer.Typer(help="Machine Health Monitoring Pipeline")


@app.command()
def run_anomaly_experiment(
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
