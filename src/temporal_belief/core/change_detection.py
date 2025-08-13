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
        self.all_change_points = []
        self.all_no_change_points = []

    def detect_persistent_changes(self, topic_timeline):
        """Detect persistent changes in stance."""

        # Convert to (utt_id, detected_stance) tuple
        # topic_timeline_list = list(topic_timeline.items())

        # Collect the tuples where the stance is persistent across n utterances
        change_points = []
        no_change_points = []

        for i in range(len(topic_timeline) - 1):
            # if current stance is different than prior
            if topic_timeline[i][1] != topic_timeline[i - 1][1]:
                # Check if change persists for more than 1 post
                if topic_timeline[i][1] == topic_timeline[i + 1][1]:
                    change_index = i
                    utt_id = topic_timeline[i][0]
                    # change_point = (change_index, utt_id)
                    change_point = (utt_id)
                    change_points.append(change_point)
                    # print(f"Current:{topic_timeline[i][1]}, Previous: {topic_timeline[i-1][1]} and Next:{topic_timeline[i+1][1]}")
                    self.all_change_points.extend(change_points)
                    self.all_no_change_points.extend(no_change_points)

        return {
            'change_points': change_points,
            'no_change_points': no_change_points
        }

    def get_two_groups(self):
        with_changes = {}
        no_changes = {}

        for user_id, topic_timelines in timelines.items():
            for topic_name, topic_timeline in topic_timelines.items():  # Added topic_name
                topic_timeline_list = list(topic_timeline.items())
                changes = self.detect_persistent_changes(topic_timeline_list)

                if changes['change_points']:
                    # User experienced changes - store only change-causing utterances
                    if user_id not in with_changes:
                        with_changes[user_id] = {}
                    with_changes[user_id][topic_name] = {utt_id: topic_timeline[utt_id] for utt_id in
                                                         changes['change_points']}
                else:
                    # User had no changes - store all utterances
                    if user_id not in no_changes:
                        no_changes[user_id] = {}
                    no_changes[user_id][topic_name] = topic_timeline

        return {
            'with_changes': with_changes,
            'no_changes': no_changes
        }

# Maybe I call the change detector to run a change analysis or something.
# What this does is it saves both the change points AND the groups in variables