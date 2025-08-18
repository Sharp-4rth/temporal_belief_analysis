import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from collections import Counter
import logging


class ChangeDetector:
    """Sliding window change detection with persistence threshold."""

    def __init__(self, window_size=3, persistence_threshold=4, significance_level=0.05):
        self.window_size = window_size
        self.persistence_threshold = persistence_threshold
        self.alpha = significance_level
        self.stance_values = {
            'strongly_against': -2, 'moderately_against': -1,
            'neutral': 0, 'moderately_favor': 1, 'strongly_favor': 2
        }
        self.all_change_points = []
        self.all_no_change_points = []

    def detect_persistent_changes(self, topic_timeline):
        """
        Detect persistent changes using sliding window with numerical averages.
        This is your main detection method.
        """
        if len(topic_timeline) < self.window_size * 2:
            return {'change_points': [], 'no_change_points': []}

        # Convert stances to numerical values
        numerical_stances = []
        for utt_id, stance in topic_timeline:
            numerical_stances.append(self.stance_values.get(stance, 0))

        change_points = []
        no_change_points = []

        # Calculate sliding window averages
        for i in range(self.window_size, len(numerical_stances) - self.window_size):

            # Get before and after windows
            before_window = numerical_stances[i - self.window_size:i]
            after_window = numerical_stances[i:i + self.window_size]

            # Calculate means
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)

            # Check for significant change (simple threshold approach)
            change_magnitude = abs(after_mean - before_mean)

            # Require both magnitude and direction consistency
            if change_magnitude > 0.5:  # Adjust threshold as needed
                # Check if change persists
                future_window = numerical_stances[i + self.window_size:i + 2 * self.window_size]
                if len(future_window) >= self.window_size:
                    future_mean = np.mean(future_window)

                    # If the change direction is maintained
                    if (after_mean - before_mean) * (future_mean - before_mean) > 0:
                        utt_id = topic_timeline[i][0]
                        change_points.append(utt_id)
                        # print(f"Sliding window change at index {i}: "
                        #       f"before={before_mean:.2f}, after={after_mean:.2f}, "
                        #       f"future={future_mean:.2f}")

        # Add non-change points
        for i, (utt_id, stance) in enumerate(topic_timeline):
            if utt_id not in change_points:
                no_change_points.append(utt_id)

        return {
            'change_points': change_points,
            'no_change_points': no_change_points
        }

    def detect_persistent_changes_simple(self, topic_timeline):
        """
        Alternative: Simple persistence detection (your original approach, but fixed).
        Call this method if you want the simpler approach.
        """
        change_points = []
        no_change_points = []

        if len(topic_timeline) < self.persistence_threshold + 1:
            # Timeline too short for meaningful analysis
            return {'change_points': change_points, 'no_change_points': no_change_points}

        # Track detected changes to avoid duplicates
        already_detected = set()

        for i in range(1, len(topic_timeline) - self.persistence_threshold):
            current_stance = topic_timeline[i][1]
            previous_stance = topic_timeline[i - 1][1]

            # Check if stance changed
            if current_stance != previous_stance and i not in already_detected:

                # Check if new stance persists for required threshold
                persistence_count = 1  # Current post counts as 1

                for j in range(i + 1, min(i + self.persistence_threshold, len(topic_timeline))):
                    if topic_timeline[j][1] == current_stance:
                        persistence_count += 1
                    else:
                        break  # Persistence broken

                # If new stance persists for threshold, mark as change point
                if persistence_count >= self.persistence_threshold:
                    utt_id = topic_timeline[i][0]
                    change_points.append(utt_id)

                    # Mark this range as detected to avoid overlapping detections
                    for k in range(i, min(i + self.persistence_threshold, len(topic_timeline))):
                        already_detected.add(k)

                    print(f"Change detected at index {i}: {previous_stance} → {current_stance} "
                          f"(persisted for {persistence_count} posts)")

        # Add non-change points (utterances that didn't cause changes)
        for i, (utt_id, stance) in enumerate(topic_timeline):
            if utt_id not in change_points:
                no_change_points.append(utt_id)

        # Store for global analysis
        self.all_change_points.extend(change_points)
        self.all_no_change_points.extend(no_change_points)

        return {
            'change_points': change_points,
            'no_change_points': no_change_points
        }

    def get_two_groups(self, timelines, method='sliding_window'):
        """
        Categorize users into those with/without changes using specified method.

        Args:
            timelines: User timeline data
            method: 'sliding_window' (default) or 'simple'
        """
        with_changes = {}
        no_changes = {}

        # Choose detection method
        if method == 'sliding_window':
            detect_func = self.detect_persistent_changes  # Uses numerical sliding windows
        elif method == 'simple':
            detect_func = self.detect_persistent_changes_simple  # Your original approach
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sliding_window' or 'simple'")

        for user_id, topic_timelines in timelines.items():
            user_has_changes = False

            for topic_name, topic_timeline in topic_timelines.items():
                topic_timeline_list = list(topic_timeline.items())
                changes = detect_func(topic_timeline_list)

                if changes['change_points']:
                    user_has_changes = True
                    # Store change-causing utterances
                    if user_id not in with_changes:
                        with_changes[user_id] = {}
                    with_changes[user_id][topic_name] = {
                        utt_id: topic_timeline[utt_id]
                        for utt_id in changes['change_points']
                    }

            # If user had no changes in any topic, add to no_changes group
            if not user_has_changes:
                no_changes[user_id] = topic_timelines

        return {
            'with_changes': with_changes,
            'no_changes': no_changes
        }