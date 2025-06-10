import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from ..utils.config import ProjectConfig


class BeliefChangeDetector:
    """Detect and analyze belief changes in user temporal data.

    This class processes user timeline data to identify significant stance
    shifts, change points, and temporal patterns in political belief evolution.

    Attributes:
        config: Project configuration with analysis parameters
        logger: Logger for tracking analysis operations
        change_threshold: Minimum change magnitude to consider significant
        window_size: Size of temporal window for change point analysis
    """

    def __init__(self, config: ProjectConfig, logger: logging.Logger):
        """Initialize belief change detector with configuration.

        Args:
            config: Project configuration containing analysis parameters
            logger: Logger instance for tracking operations
        """
        self.config = config
        self.logger = logger
        self.change_threshold = config.change_threshold
        self.window_size = config.temporal_window_days

    def detect_belief_changes(self, user_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect significant belief changes in user's temporal posting history.

        Analyzes stance trajectory to identify points where user's political
        position shifted significantly, with statistical validation.

        Args:
            user_timeline: List of posts with timestamps and stance predictions

        Returns:
            List of detected change points with metadata

        Example:
            >>> timeline = [{'timestamp': date, 'stance': 'liberal', 'confidence': 0.9}, ...]
            >>> changes = detector.detect_belief_changes(timeline)
            >>> print(f"Detected {len(changes)} significant belief changes")
        """
        if len(user_timeline) < self.window_size:
            self.logger.warning(f"Timeline too short for analysis: {len(user_timeline)} posts")
            return []

        changes = []
        stance_scores = self._convert_to_numeric_scores(user_timeline)

        # Sliding window change point detection
        for i in range(self.window_size, len(stance_scores) - self.window_size):
            before_window = stance_scores[i - self.window_size:i]
            after_window = stance_scores[i:i + self.window_size]

            # Calculate change magnitude
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)
            change_magnitude = abs(after_mean - before_mean)

            # Statistical significance test
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(before_window, after_window)

            if (change_magnitude > self.change_threshold and
                    p_value < self.config.significance_level):
                change_point = {
                    'timestamp': user_timeline[i]['timestamp'],
                    'index': i,
                    'from_stance_mean': before_mean,
                    'to_stance_mean': after_mean,
                    'change_magnitude': change_magnitude,
                    'p_value': p_value,
                    'confidence': user_timeline[i]['confidence'],
                    'direction': 'progressive' if after_mean > before_mean else 'conservative'
                }
                changes.append(change_point)

                self.logger.debug(f"Change detected at {change_point['timestamp']}: "
                                  f"{change_magnitude:.3f} magnitude, p={p_value:.3f}")

        self.logger.info(f"Detected {len(changes)} significant belief changes")
        return changes

    def _convert_to_numeric_scores(self, timeline: List[Dict[str, Any]]) -> List[float]:
        """Convert stance labels to numeric scores for analysis.

        Args:
            timeline: List of posts with stance predictions

        Returns:
            List of numeric stance scores (-1 to +1 scale)
        """
        stance_mapping = {'against': -1.0, 'neutral': 0.0, 'favor': 1.0}
        return [stance_mapping.get(post['stance'], 0.0) for post in timeline]

    def analyze_temporal_patterns(self, user_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in user's belief evolution.

        Args:
            user_timeline: User's posting history with stance predictions

        Returns:
            Dictionary containing temporal pattern analysis results
        """
        if not user_timeline:
            return {}

        stance_scores = self._convert_to_numeric_scores(user_timeline)
        timestamps = [post['timestamp'] for post in user_timeline]

        # Calculate temporal statistics
        analysis = {
            'total_posts': len(user_timeline),
            'time_span_days': (max(timestamps) - min(timestamps)).days,
            'stance_variance': np.var(stance_scores),
            'stance_trend': np.corrcoef(range(len(stance_scores)), stance_scores)[0, 1],
            'average_confidence': np.mean([post['confidence'] for post in user_timeline]),
            'posting_frequency': len(user_timeline) / max(1, (max(timestamps) - min(timestamps)).days),
            'stance_consistency': 1.0 - np.var(stance_scores)  # Higher = more consistent
        }

        return analysis