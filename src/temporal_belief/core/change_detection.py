import numpy as np
from collections import Counter
import logging
from typing import Dict, List, Tuple, Any, Optional


class ChangeDetector:
    """CUSUM-based change detection for political stance shifts.

    Focuses on detecting changes between 'left-leaning' and 'right-leaning' positions,
    ignoring neutral stances. Uses cumulative sum control charts to identify
    significant shifts in political orientation over time.
    """

    def __init__(self, threshold=6.0, drift=0.5, min_change_separation=5):
        """Initialize CUSUM detector with control parameters.

        Args:
            threshold: Detection threshold for CUSUM statistic (higher = less sensitive)
            drift: Reference drift value for change detection (typically 0.5-1.0)
            min_change_separation: Minimum posts between detected changes
        """
        self.threshold = threshold
        self.drift = drift
        self.min_change_separation = min_change_separation

        # Map stances to numeric values for CUSUM
        self.stance_values = {
            'left-leaning': -1.0,
            'neutral': 0.0,  # Will be filtered out
            'right-leaning': 1.0
        }

        self.all_change_points = []
        self.all_no_change_points = []

        # Logging setup
        self.logger = logging.getLogger(__name__)

    def _to_probs(self, item):
        """Convert various input formats to probability tuple (pL, pN, pR)."""
        if isinstance(item, str):
            if item == 'left-leaning':  return (1.0, 0.0, 0.0)
            if item == 'neutral':       return (0.0, 1.0, 0.0)
            if item == 'right-leaning': return (0.0, 0.0, 1.0)
            return (0.0, 1.0, 0.0)
        if isinstance(item, dict):
            return (float(item.get('pL', 0.0)), float(item.get('pN', 0.0)), float(item.get('pR', 0.0)))
        if isinstance(item, (list, tuple)) and len(item) == 3:
            pL, pN, pR = item
            return (float(pL), float(pN), float(pR))
        return (0.0, 1.0, 0.0)

    def _get_political_signal(self, prob_tuple, conf_threshold=0.6):
        """Extract political signal from probability tuple, ignoring neutral.

        Args:
            prob_tuple: (pL, pN, pR) probability tuple
            conf_threshold: Minimum confidence to consider stance reliable

        Returns:
            Float value: -1.0 (left), +1.0 (right), or None (neutral/uncertain)
        """
        pL, pN, pR = prob_tuple

        # Only consider if we have sufficient confidence in left or right
        if pL >= conf_threshold:
            return -1.0  # left-leaning
        elif pR >= conf_threshold:
            return 1.0  # right-leaning
        else:
            return None  # neutral or uncertain - ignore for CUSUM

    def detect_cusum_changes_advanced(self, topic_timeline, conf_threshold=0.6,
                                      adaptive_threshold=True):
        """Advanced CUSUM with adaptive thresholding and confidence weighting.

        Args:
            topic_timeline: List of (utterance_id, stance_data) tuples
            conf_threshold: Minimum confidence for reliable stance detection
            adaptive_threshold: Whether to adapt threshold based on signal variance

        Returns:
            Dictionary with change_points and no_change_points lists
        """
        if not topic_timeline:
            return {'change_points': [], 'no_change_points': []}

        # Extract weighted political signals
        signals = []
        confidences = []
        valid_utterances = []

        for utt_id, stance_data in topic_timeline:
            prob_tuple = self._to_probs(stance_data)
            signal = self._get_political_signal(prob_tuple, conf_threshold)

            if signal is not None:
                signals.append(signal)
                # Extract confidence from stance_data if available
                confidence = self._extract_confidence(stance_data)
                confidences.append(confidence)
                valid_utterances.append(utt_id)

        if len(signals) < 3:
            return {'change_points': [], 'no_change_points': [utt_id for utt_id, _ in topic_timeline]}

        # Adaptive threshold based on signal variance
        threshold = self.threshold
        if adaptive_threshold:
            signal_std = np.std(signals)
            threshold = max(self.threshold, 2.0 * signal_std)
            self.logger.debug(f"CUSUM: Adaptive threshold set to {threshold:.2f}")

        # Confidence-weighted CUSUM
        change_indices = self._cusum_detect_changes_weighted(signals, confidences, threshold)

        change_points = [valid_utterances[idx] for idx in change_indices if idx < len(valid_utterances)]
        change_set = set(change_points)
        no_change_points = [utt_id for utt_id, _ in topic_timeline if utt_id not in change_set]

        self.all_change_points.extend(change_points)
        self.all_no_change_points.extend(no_change_points)

        return {
            'change_points': change_points,
            'no_change_points': no_change_points
        }

    def _cusum_detect_changes_weighted(self, signals, confidences, threshold):
        """CUSUM with confidence weighting for more reliable change detection."""
        signals = np.array(signals)
        confidences = np.array(confidences)
        n = len(signals)
        change_points = []

        # Confidence-weighted mean
        weighted_mean = np.average(signals, weights=confidences)

        # Initialize CUSUM with confidence weighting
        cusum_pos = 0.0
        cusum_neg = 0.0

        for i in range(1, n):
            # Weight deviation by confidence
            deviation = (signals[i] - weighted_mean) * confidences[i]

            # Update CUSUM statistics
            cusum_pos = max(0, cusum_pos + deviation - self.drift)
            cusum_neg = max(0, cusum_neg - deviation - self.drift)

            # Detection with separation enforcement
            if cusum_pos > threshold or cusum_neg > threshold:
                if not change_points or i - change_points[-1] >= self.min_change_separation:
                    change_points.append(i)
                    cusum_pos = 0.0
                    cusum_neg = 0.0

                    direction = "right" if cusum_pos > cusum_neg else "left"
                    self.logger.debug(
                        f"CUSUM: {direction} shift detected at index {i}, confidence={confidences[i]:.2f}")

        return change_points

    def _extract_confidence(self, stance_data):
        """Extract confidence score from stance data."""
        if isinstance(stance_data, dict):
            return stance_data.get('confidence', 1.0)
        elif isinstance(stance_data, (list, tuple)) and len(stance_data) == 3:
            # Use max probability as confidence
            return max(stance_data)
        else:
            return 1.0  # Default confidence

    def get_two_groups(self, timelines, method='cusum', conf_threshold=0.6,
                       advanced=True, **kwargs):
        """
        Group users into with/without changes using CUSUM detection.

        Args:
            timelines: Dictionary of {user_id: {topic: timeline}} data
            method: Detection method ('cusum' or 'cusum_advanced')
            conf_threshold: Minimum confidence for reliable stance detection
            advanced: Whether to use confidence-weighted CUSUM
            **kwargs: Additional parameters (threshold, drift, etc.)

        Returns:
            Dictionary with 'with_changes' and 'no_changes' user groups
        """
        with_changes = {}
        no_changes = {}

        # Update detector parameters from kwargs
        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        if 'drift' in kwargs:
            self.drift = kwargs['drift']
        if 'min_change_separation' in kwargs:
            self.min_change_separation = kwargs['min_change_separation']

        # Select detection method
        if advanced:
            detect_func = lambda tl: self.detect_cusum_changes_advanced(
                tl, conf_threshold=conf_threshold, **kwargs
            )
        else:
            detect_func = lambda tl: self.detect_cusum_changes(
                tl, conf_threshold=conf_threshold
            )

        self.logger.info(f"Starting CUSUM change detection with threshold={self.threshold}, "
                         f"drift={self.drift}, advanced={advanced}")

        for user_id, topic_timelines in timelines.items():
            if user_id == '[deleted]':
                continue

            user_has_changes = False

            for topic_name, topic_timeline in topic_timelines.items():
                # Convert to list format expected by detection methods
                topic_timeline_list = list(topic_timeline.items())

                # Run CUSUM change detection
                changes = detect_func(topic_timeline_list)

                if changes['change_points']:
                    user_has_changes = True
                    if user_id not in with_changes:
                        with_changes[user_id] = {}

                    # Store change points with their stance data
                    with_changes[user_id][topic_name] = {
                        utt_id: topic_timeline[utt_id]
                        for utt_id in changes['change_points']
                    }

            # Users without any detected changes
            if not user_has_changes:
                no_changes[user_id] = topic_timelines

        # Log summary statistics
        self.logger.info(f"CUSUM Results: {len(with_changes)} users with changes, "
                         f"{len(no_changes)} users without changes")
        self.logger.info(f"Total change points detected: {len(self.all_change_points)}")

        return {
            'with_changes': with_changes,
            'no_changes': no_changes,
            'summary': {
                'users_with_changes': len(with_changes),
                'users_without_changes': len(no_changes),
                'total_change_points': len(self.all_change_points),
                'detection_parameters': {
                    'threshold': self.threshold,
                    'drift': self.drift,
                    'min_separation': self.min_change_separation,
                    'conf_threshold': conf_threshold
                }
            }
        }