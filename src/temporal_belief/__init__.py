"""
Temporal Belief Evolution Analysis Package
To be written.
"""

__version__ = "1.0.0"
__author__ = "Leonidas Theodoropoulos"
__description__ = "Temporal political stance detection and belief change prediction"

from .utils.config import ProjectConfig
from .utils.logger import setup_logging
from data.loaders import RedditDataLoader
from .models.stance_detection_models import StanceDetector
from .core.temporal_analyzer import TemporalAnalyzer

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
    'StanceDetector'
    'TemporalAnalyzer',
    'RedditDataLoader',
    'initialize_project'
]