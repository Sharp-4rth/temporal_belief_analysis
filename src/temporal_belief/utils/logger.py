import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure comprehensive logging for the entire project.

    Creates a logger with both file and console handlers, formatted for
    easy debugging and monitoring of training progress.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    Returns:
        Configured logger instance
    Example:
        >>> logger = setup_logging("DEBUG")
        >>> logger.info("Starting stance detection training")
    """
    logger = logging.getLogger("temporal_belief")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # File handler for persistent logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"belief_analysis_{timestamp}.log")

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()

    # Detailed formatter for comprehensive context
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger