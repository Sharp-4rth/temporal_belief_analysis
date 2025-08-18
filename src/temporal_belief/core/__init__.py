"""Core analysis components."""

from .topic_detection import TopicDetector
from .stance_detection_old import StanceDetector

__all__ = [
    "TopicDetector",
    "StanceDetector"
]