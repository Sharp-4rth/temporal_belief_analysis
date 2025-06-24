import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import torch


@dataclass
class ProjectConfig:
    """Central configuration for the entire project.

    This class manages all configuration parameters, file paths, and model
    hyperparameters in one place for easy maintenance and reproducibility.

    Attributes:
        project_name: Name of the project for logging and outputs
        version: Current project version for tracking
        data_dir: Root directory for all data files
        model_dir: Directory for saved models and checkpoints
        results_dir: Directory for output results and reports
        bert_model_name: Pre-trained BERT model identifier
        lstm_hidden_size: Hidden dimension size for LSTM networks
        max_sequence_length: Maximum input sequence length for models
        batch_size: Training and inference batch size
        learning_rate: Learning rate for model training
        confidence_threshold: Minimum confidence for reliable predictions
        random_seed: Random seed for reproducibility
    """
    # Project metadata
    project_name: str = "temporal-belief-analysis"
    version: str = "1.0.0"

    # Directory structure
    data_dir: Path = Path("data")
    model_dir: Path = Path("models/saved")
    results_dir: Path = Path("reports/results")
    figures_dir: Path = Path("reports/figures")

    # Model configurations
    bert_model_name: str = "bert-base-uncased"
    lstm_hidden_size: int = 128
    max_sequence_length: int = 512
    temporal_window_days: int = 30
    prediction_horizon_days: int = 7

    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    confidence_threshold: float = 0.8

    # Analysis parameters
    event_window_days: int = 7
    change_threshold: float = 0.7
    significance_level: float = 0.05

    # System configuration
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Create necessary directories after initialization."""
        for directory in [self.data_dir, self.model_dir, self.results_dir, self.figures_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# Classification labels - separate from the main config class
POLITICAL_TOPICS = [
    "healthcare",
    "economy",
    "immigration",
    "environment",
    "education",
    "foreign_policy",
    "social_issues",
    "taxation",
    "gun_rights",
    "abortion",
    "other"
]

STANCE_LABELS = [
    "liberal",
    "conservative",
    "neutral"
]

DETAILED_STANCE_LABELS = [
    "strongly liberal",
    "moderately liberal",
    "neutral",
    "moderately conservative",
    "strongly conservative"
]