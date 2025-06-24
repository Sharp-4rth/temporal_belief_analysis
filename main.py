"""
Main entry point for Temporal Belief Evolution Analysis.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from temporal_belief.poc.stance_detection import StanceDetector


def run_analysis():
    """Run the complete temporal belief analysis pipeline."""
    pass

if __name__ == "__main__":
    run_analysis()