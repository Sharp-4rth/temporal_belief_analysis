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

    def detect_persistent_changes(self, topic_timeline):
        """Detect persistent changes in stance."""

        # Convert to (utt_id, detected_stance) tuple
        # topic_timeline_list = list(topic_timeline.items())

        # Collect the tuples where the stance is persistent across n utterances
        change_points = []

        for i in range(len(topic_timeline)):
          # if current stance is different than prior
          if topic_timeline[i][1] != topic_timeline[i-1][1]:
            # Check if change persists for more than 1 post
            if topic_timeline[i][1] == topic_timeline[i+1][1]:
              change_index = i
              utt_id = topic_timeline[i][0]
              change_point = (change_index, utt_id)
              change_points.append(change_point)
              # print(f"Current:{topic_timeline[i][1]}, Previous: {topic_timeline[i-1][1]} and Next:{topic_timeline[i+1][1]}")

        return change_points

