"""
Main entry point for Temporal Belief Evolution Analysis.
File: main.py (at project root)

This is your primary script to run the complete analysis pipeline.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from temporal_belief.models.stance_detection_models import StanceDetector
from temporal_belief.data.loaders import RedditDataLoader
from temporal_belief.core.temporal_analyzer import TemporalAnalyzer
from temporal_belief.utils.config import ProjectConfig
from temporal_belief.utils.logger import setup_logging

def run_analysis():
    """Run the complete temporal belief analysis pipeline."""
    pass

if __name__ == "__main__":
    run_analysis()