import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from collections import Counter
import logging

class ChangeDetector:
    """Sliding window change detection with proper statistical significance."""

    def __init__(self, window_size=3, significance_level=0.05):
        self.window_size = window_size
        self.alpha = significance_level
        self.stance_values = {
            'strongly_against': -2, 'moderately_against': -1,
            'neutral': 0, 'moderately_favor': 1, 'strongly_favor': 2
        }

    def detect_simple_stance_changes(self, topic_timeline):

        if len(topic_timeline) < 2:
            return []

        changes = []
        timeline_items = list(topic_timeline.items())  # Convert to list of (utterance_id, stance) pairs

        for i in range(1, len(timeline_items)):
            current_utterance_id, current_stance = timeline_items[i]
            previous_utterance_id, previous_stance = timeline_items[i-1]

            # Check if stance changed
            if current_stance != previous_stance:
                change = {
                    'position': i,
                    'current_utterance_id': current_utterance_id,
                    'previous_utterance_id': previous_utterance_id,
                    'from_stance': previous_stance,
                    'to_stance': current_stance,
                    'change_type': self._classify_change_direction(previous_stance, current_stance),
                    'change_magnitude': self._calculate_simple_magnitude(previous_stance, current_stance)
                }
                changes.append(change)

        return changes

    def _classify_change_direction(self, from_stance, to_stance):
        """Classify the direction of stance change."""
        from_value = self.stance_values.get(from_stance, 0)
        to_value = self.stance_values.get(to_stance, 0)

        if to_value > from_value:
            return 'more_favorable'
        elif to_value < from_value:
            return 'less_favorable'
        else:
            return 'neutral_shift'

    def _calculate_simple_magnitude(self, from_stance, to_stance):
        """Calculate the magnitude of stance change."""
        from_value = self.stance_values.get(from_stance, 0)
        to_value = self.stance_values.get(to_stance, 0)
        return abs(to_value - from_value)

    def detect_changes_with_significance(self, topic_timeline):
        """Detect changes with statistical significance testing."""

        if len(topic_timeline) < self.window_size * 2:
            return [], [], []

        # Convert to lists to maintain order and get IDs
        timeline_items = list(topic_timeline.items())  # [(utterance_id, stance), ...]
        stance_sequence = [self.stance_values.get(stance, 0) for _, stance in timeline_items]

        potential_changes = []
        p_values = []

        # Sliding window approach
        for i in range(self.window_size, len(stance_sequence) - self.window_size):

            # Left window (before potential change)
            left_window = stance_sequence[i - self.window_size:i]

            # Right window (after potential change)
            right_window = stance_sequence[i:i + self.window_size]

            # Statistical test: Are these two windows significantly different?
            statistic, p_value = self.two_sample_test(left_window, right_window)

            p_values.append(p_value)

            # Store potential change info with just the key utterance ID
            change_magnitude = abs(np.mean(right_window) - np.mean(left_window))
            potential_changes.append({
                'position': i,
                'utterance_id': timeline_items[i][0],  # The utterance where change detected
                'p_value': p_value,
                'test_statistic': statistic,
                'magnitude': change_magnitude,
                'left_mean': np.mean(left_window),
                'right_mean': np.mean(right_window),
                'left_window': left_window.copy(),
                'right_window': right_window.copy()
            })

        # Apply FDR correction to all p-values
        if not p_values:
            return [], [], []

        rejected, p_corrected = self.multiple_testing_correction(p_values)

        # Keep only changes that survive FDR correction
        significant_changes = []
        for i, change in enumerate(potential_changes):
            if rejected[i]:  # Survives FDR correction
                change.update({
                    'p_corrected': p_corrected[i],
                    'statistically_significant': True,
                    'survives_fdr_correction': True,
                    'significance_level': self.alpha
                })
                significant_changes.append(change)

        return significant_changes, p_values, p_corrected

    def two_sample_test(self, left_window, right_window):
        """Statistical test for difference between two windows."""
        # Use Mann-Whitney U test (non-parametric, more robust)
        try:
            statistic, p_value = mannwhitneyu(left_window, right_window,
                                            alternative='two-sided')
            return statistic, p_value
        except ValueError:
            # Fallback to t-test if Mann-Whitney fails
            statistic, p_value = ttest_ind(left_window, right_window)
            return statistic, p_value

    def multiple_testing_correction(self, p_values):
        """Correct for multiple testing using Benjamini-Hochberg."""
        rejected, p_corrected = fdrcorrection(p_values, alpha=self.alpha)
        return rejected, p_corrected

    # def analyze_user_belief_changes(self, user_timeline):
    #     """Analyze belief changes across all topics for a user."""
    #     all_changes = {}
    #
    #     for topic, topic_timeline in user_timeline.items():
    #         changes = self.detect_changes_with_significance(topic_timeline)
    #         all_changes[topic] = changes
    #
    #     return all_changes

    def analyze_user_belief_changes(self, user_timeline):
        """Analyze belief changes across all topics for a user.

        Args:
            user_timeline: Dict of {topic: {utterance_id: stance}}

        Returns:
            Dict with changes by topic and total count
        """
        all_changes = {}
        total_changes = 0

        for topic, topic_timeline in user_timeline.items():
            significant_changes, p_values, p_corrected = self.detect_changes_with_significance(topic_timeline)
            all_changes[topic] = significant_changes
            total_changes += len(significant_changes)

        return {
            'changes_by_topic': all_changes,
            'total_changes': total_changes
        }

    def analyze_all_users_belief_changes(self, timelines):
        """Analyze belief changes across all users.

        Args:
            timelines: Dict of {user_id: {topic: {utterance_id: stance}}}

        Returns:
            Dict with changes by user and total count
        """
        all_user_changes = {}
        total_changes = 0

        for user_id, user_timeline in timelines.items():
            user_result = self.analyze_user_belief_changes(user_timeline)
            all_user_changes[user_id] = user_result
            total_changes += user_result['total_changes']

        return {
            'changes_by_user': all_user_changes,
            'total_changes': total_changes
        }