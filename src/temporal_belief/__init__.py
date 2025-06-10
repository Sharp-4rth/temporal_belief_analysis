"""
Temporal Belief Evolution Analysis Package

A comprehensive system for analyzing and predicting political belief changes
over time using BERT and LSTM models.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main classes for easy access
from .utils.config import ProjectConfig
from .utils.logger import setup_logging
from .core.stance_detector import StanceDetector
from .core.temporal_analyzer import BeliefChangeDetector


# Package-level convenience function
def initialize_project(log_level: str = "INFO"):
    """Initialize project with configuration and logging.

    Args:
        log_level: Logging level for the session

    Returns:
        Tuple of (config, logger) for immediate use
    """
    config = ProjectConfig()
    logger = setup_logging(log_level)

    logger.info(f"Initialized {config.project_name} v{config.version}")
    logger.info(f"Using device: {config.device}")

    return config, logger


__all__ = [
    'ProjectConfig',
    'setup_logging',
    'StanceDetector',
    'BeliefChangeDetector',
    'initialize_project'
]