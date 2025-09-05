"""Configuration builder for anomaly detection models."""

import os
from typing import Dict

import pandas as pd
import yaml

from machine_health.config.training_config import TrainingConfig, FEATURES
from utils.logging import logger


class ConfigBuilder:
    """Handles configuration file generation with baselines and limits."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_cols = FEATURES
    
    def calculate_baselines(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline values for each signal type."""
        # Calculate mean for numeric columns only
        baselines = df[self.feature_cols].mean().to_dict()
        # Convert numpy types to regular Python types for YAML serialization
        baselines = {k: float(v) for k, v in baselines.items()}
        logger.info(f"Baselines: {baselines}")
        
        return baselines
    
    def calculate_allowed_extremes(self, df: pd.DataFrame) -> Dict[str, list]:
        """Calculate allowed extremes (3-sigma limits) for each signal."""
        allowed_extremes = {}
        
        # Process current, power factor, voltage, and THD signals
        for signal in self.feature_cols:
            mean, std = float(df[signal].mean()), float(df[signal].std())
            allowed_extremes[f"{signal}"] = [mean - 3*std, mean + 3*std]
        
        return allowed_extremes
    
    def build_config(self, df: pd.DataFrame, threshold: float) -> Dict:
        """Build complete configuration dictionary."""
        logger.info("Building configuration file with signal baselines and limits...")
        
        try:
            baselines = self.calculate_baselines(df)
            allowed_extremes = self.calculate_allowed_extremes(df)
            
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
