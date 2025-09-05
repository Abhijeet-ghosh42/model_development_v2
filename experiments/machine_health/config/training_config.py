"""Training configuration for anomaly detection models."""

from dataclasses import dataclass
from typing import List, Optional

# Constants
DATA_DIR = "machine_health/data"
ARTIFACTS_DIR = "machine_health/artifacts"
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
