"""
Simple temporal analysis for detecting belief changes.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta


class TemporalAnalyzer:
    """Simple analyzer for detecting belief changes over time."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        # Simple parameters
        self.change_threshold = 0.5  # How big a change to consider significant
        self.window_size = 5  # How many posts to look at for change detection

    def detect_belief_changes(self, user_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect significant belief changes in a user's timeline.

        Args:
            user_timeline: List of posts with stance predictions

        Returns:
            List of detected change points
        """
        if len(user_timeline) < self.window_size * 2:
            self.logger.debug(f"Timeline too short: {len(user_timeline)} posts")
            return []

        changes = []
        stance_scores = self._convert_to_scores(user_timeline)

        # Simple sliding window change detection
        for i in range(self.window_size, len(stance_scores) - self.window_size):
            # Compare before and after windows
            before_window = stance_scores[i - self.window_size:i]
            after_window = stance_scores[i:i + self.window_size]

            before_avg = np.mean(before_window)
            after_avg = np.mean(after_window)

            change_magnitude = abs(after_avg - before_avg)

            # Check if change is significant
            if change_magnitude > self.change_threshold:
                change_point = {
                    'timestamp': user_timeline[i]['timestamp'],
                    'index': i,
                    'from_stance_avg': before_avg,
                    'to_stance_avg': after_avg,
                    'change_magnitude': change_magnitude,
                    'direction': 'more_liberal' if after_avg > before_avg else 'more_conservative'
                }
                changes.append(change_point)

                self.logger.debug(f"Change detected at index {i}: {change_magnitude:.2f}")

        self.logger.info(f"Detected {len(changes)} belief changes in timeline")
        return changes

    def _convert_to_scores(self, timeline: List[Dict[str, Any]]) -> List[float]:
        """
        Convert stance labels to numeric scores for analysis.

        against = -1, neutral = 0, favor = +1
        """
        stance_mapping = {'against': -1.0, 'neutral': 0.0, 'favor': 1.0}

        scores = []
        for post in timeline:
            stance = post.get('stance', 'neutral')
            score = stance_mapping.get(stance, 0.0)
            scores.append(score)

        return scores

    def analyze_user_patterns(self, user_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze temporal patterns for a single user.

        Args:
            user_timeline: User's posts with stance predictions

        Returns:
            Dictionary with pattern analysis
        """
        if not user_timeline:
            return {}

        stance_scores = self._convert_to_scores(user_timeline)
        timestamps = [post['timestamp'] for post in user_timeline]

        # Calculate basic statistics
        analysis = {
            'total_posts': len(user_timeline),
            'time_span_days': (max(timestamps) - min(timestamps)).days,
            'stance_variance': np.var(stance_scores),
            'average_stance': np.mean(stance_scores),
            'stance_trend': self._calculate_trend(stance_scores),
            'consistency': 1.0 - np.var(stance_scores),  # Higher = more consistent
            'posting_frequency': len(user_timeline) / max(1, (max(timestamps) - min(timestamps)).days)
        }

        return analysis

    def _calculate_trend(self, scores: List[float]) -> float:
        """Calculate overall trend in stance scores (positive = becoming more liberal)."""
        if len(scores) < 2:
            return 0.0

        # Simple linear trend
        x = np.arange(len(scores))
        correlation = np.corrcoef(x, scores)[0, 1]

        return correlation if not np.isnan(correlation) else 0.0

    def batch_analyze_users(self, user_timelines: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Analyze patterns for multiple users.

        Args:
            user_timelines: Dict mapping user_id to their timeline

        Returns:
            Dict mapping user_id to their analysis results
        """
        results = {}

        self.logger.info(f"Analyzing patterns for {len(user_timelines)} users...")

        for user_id, timeline in user_timelines.items():
            # Analyze patterns
            patterns = self.analyze_user_patterns(timeline)

            # Detect changes
            changes = self.detect_belief_changes(timeline)

            results[user_id] = {
                'patterns': patterns,
                'changes': changes,
                'num_changes': len(changes)
            }

        self.logger.info(f"Completed analysis for {len(results)} users")
        return results


def quick_test():
    """Test the temporal analyzer."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    analyzer = TemporalAnalyzer(logger)

    # Create sample timeline with stance predictions
    sample_timeline = [
        {'timestamp': datetime(2024, 1, 1), 'stance': 'favor', 'text': 'Liberal post 1'},
        {'timestamp': datetime(2024, 1, 5), 'stance': 'favor', 'text': 'Liberal post 2'},
        {'timestamp': datetime(2024, 1, 10), 'stance': 'neutral', 'text': 'Neutral post'},
        {'timestamp': datetime(2024, 1, 15), 'stance': 'against', 'text': 'Conservative post 1'},
        {'timestamp': datetime(2024, 1, 20), 'stance': 'against', 'text': 'Conservative post 2'},
        {'timestamp': datetime(2024, 1, 25), 'stance': 'against', 'text': 'Conservative post 3'},
        {'timestamp': datetime(2024, 2, 1), 'stance': 'neutral', 'text': 'Back to neutral'},
        {'timestamp': datetime(2024, 2, 5), 'stance': 'favor', 'text': 'Back to liberal'},
    ]

    print("Testing Temporal Analysis:")
    print("=" * 50)

    # Analyze user patterns
    patterns = analyzer.analyze_user_patterns(sample_timeline)
    print("User Patterns:")
    for key, value in patterns.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    print()

    # Detect changes
    changes = analyzer.detect_belief_changes(sample_timeline)
    print(f"Detected {len(changes)} belief changes:")
    for i, change in enumerate(changes):
        print(f"  {i + 1}. {change['timestamp']} - {change['direction']} (magnitude: {change['change_magnitude']:.2f})")

    return analyzer


if __name__ == "__main__":
    analyzer = quick_test()