import numpy as np
from collections import Counter
import logging
from typing import Dict, List, Tuple, Any, Optional
logging.disable(logging.CRITICAL)

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

    def detect_cusum_changes(self, topic_timeline, conf_threshold=0.6):
        """Detect political stance changes using CUSUM algorithm.

        Args:
            topic_timeline: List of (utterance_id, stance_data) tuples
            conf_threshold: Minimum confidence for reliable stance detection

        Returns:
            Dictionary with change_points and no_change_points lists
        """
        if not topic_timeline:
            return {'change_points': [], 'no_change_points': []}

        # Extract political signals, filtering out neutral/uncertain
        signals = []
        valid_utterances = []

        for utt_id, stance_data in topic_timeline:
            prob_tuple = self._to_probs(stance_data)
            signal = self._get_political_signal(prob_tuple, conf_threshold)

            if signal is not None:
                signals.append(signal)
                valid_utterances.append(utt_id)

        if len(signals) < 3:
            self.logger.warning(f"Insufficient political signals for CUSUM: {len(signals)}")
            return {'change_points': [], 'no_change_points': [utt_id for utt_id, _ in topic_timeline]}

        # CUSUM change detection
        change_indices = self._cusum_detect_changes(signals)

        # Convert indices back to utterance IDs
        change_points = [valid_utterances[idx] for idx in change_indices if idx < len(valid_utterances)]

        # All other utterances are no-change points
        change_set = set(change_points)
        no_change_points = [utt_id for utt_id, _ in topic_timeline if utt_id not in change_set]

        # Store for aggregate statistics
        self.all_change_points.extend(change_points)
        self.all_no_change_points.extend(no_change_points)

        return {
            'change_points': change_points,
            'no_change_points': no_change_points
        }

    def _cusum_detect_changes(self, signals):
        """Core CUSUM algorithm for detecting mean shifts in political stance.

        Args:
            signals: List of political stance values (-1.0 or +1.0)

        Returns:
            List of indices where significant changes were detected
        """
        if len(signals) < 2:
            return []

        signals = np.array(signals)
        n = len(signals)
        change_points = []

        # Calculate overall mean for reference
        overall_mean = np.mean(signals)

        # Initialize CUSUM statistics
        cusum_pos = 0.0  # Positive CUSUM (detecting upward shifts)
        cusum_neg = 0.0  # Negative CUSUM (detecting downward shifts)

        for i in range(1, n):
            # Calculate deviations from reference mean
            deviation = signals[i] - overall_mean

            # Update CUSUM statistics
            cusum_pos = max(0, cusum_pos + deviation - self.drift)
            cusum_neg = max(0, cusum_neg - deviation - self.drift)

            # Check for threshold crossings
            change_detected = False

            if cusum_pos > self.threshold:
                # Positive shift detected (towards right-leaning)
                change_points.append(i)
                cusum_pos = 0.0  # Reset after detection
                change_detected = True
                self.logger.debug(f"CUSUM: Positive shift detected at index {i}")

            elif cusum_neg > self.threshold:
                # Negative shift detected (towards left-leaning)
                change_points.append(i)
                cusum_neg = 0.0  # Reset after detection
                change_detected = True
                self.logger.debug(f"CUSUM: Negative shift detected at index {i}")

            # Enforce minimum separation between changes
            if change_detected and len(change_points) > 1:
                if i - change_points[-2] < self.min_change_separation:
                    change_points.pop()  # Remove this change point
                    self.logger.debug(f"CUSUM: Removed change point at {i} due to minimum separation")

        return change_points

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

    def _get_political_signal(self, prob_tuple, conf_threshold=0.6):
        """Extract political signal, ignoring neutral positions."""
        pL, pN, pR = prob_tuple

        # Only consider confident left/right positions
        if pL >= conf_threshold:
            return -1.0  # left-leaning
        elif pR >= conf_threshold:
            return 1.0  # right-leaning
        else:
            return None  # neutral/uncertain - ignore

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

    def analyze_change_patterns(self, with_changes_data):
        """Analyze patterns in detected political stance changes.

        Args:
            with_changes_data: Users with detected changes from get_two_groups()

        Returns:
            Dictionary containing change pattern analysis
        """
        all_changes = []

        for user_id, topics in with_changes_data.items():
            for topic_name, change_points in topics.items():
                for utt_id, stance_data in change_points.items():
                    prob_tuple = self._to_probs(stance_data)
                    signal = self._get_political_signal(prob_tuple)

                    if signal is not None:
                        all_changes.append({
                            'user_id': user_id,
                            'topic': topic_name,
                            'utterance_id': utt_id,
                            'direction': 'left_shift' if signal < 0 else 'right_shift',
                            'magnitude': abs(signal),
                            'confidence': self._extract_confidence(stance_data)
                        })

        if not all_changes:
            return {'total_changes': 0}

        # Analyze patterns
        change_directions = [c['direction'] for c in all_changes]
        change_magnitudes = [c['magnitude'] for c in all_changes]
        change_confidences = [c['confidence'] for c in all_changes]

        direction_counts = Counter(change_directions)

        return {
            'total_changes': len(all_changes),
            'direction_distribution': dict(direction_counts),
            'average_magnitude': np.mean(change_magnitudes),
            'average_confidence': np.mean(change_confidences),
            'left_shifts': direction_counts.get('left_shift', 0),
            'right_shifts': direction_counts.get('right_shift', 0),
            'most_common_direction': direction_counts.most_common(1)[0] if direction_counts else None
        }

    def tune_cusum_parameters(self, validation_timeline, known_changes=None):
        """Tune CUSUM parameters for optimal performance on validation data.

        Args:
            validation_timeline: Timeline with known change points for tuning
            known_changes: List of known change points for comparison

        Returns:
            Dictionary with optimal parameters and performance metrics
        """
        # Parameter grid for tuning
        threshold_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        drift_values = [0.3, 0.5, 0.7, 1.0]

        best_params = None
        best_score = -1.0
        results = []

        for threshold in threshold_values:
            for drift in drift_values:
                # Temporarily set parameters
                original_threshold = self.threshold
                original_drift = self.drift

                self.threshold = threshold
                self.drift = drift

                # Test detection
                detected = self.detect_cusum_changes(validation_timeline)

                # Calculate performance metrics
                if known_changes:
                    precision, recall, f1 = self._calculate_detection_metrics(
                        detected['change_points'], known_changes
                    )
                    score = f1
                else:
                    # Use change detection rate as proxy metric
                    score = len(detected['change_points']) / max(1, len(validation_timeline))

                results.append({
                    'threshold': threshold,
                    'drift': drift,
                    'score': score,
                    'change_points': len(detected['change_points'])
                })

                if score > best_score:
                    best_score = score
                    best_params = {'threshold': threshold, 'drift': drift}

                # Restore original parameters
                self.threshold = original_threshold
                self.drift = original_drift

        # Set best parameters
        if best_params:
            self.threshold = best_params['threshold']
            self.drift = best_params['drift']

        self.logger.info(f"CUSUM tuning complete. Best params: {best_params}, Score: {best_score:.3f}")

        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results
        }

    def _calculate_detection_metrics(self, detected_changes, known_changes):
        """Calculate precision, recall, and F1 for change detection."""
        detected_set = set(detected_changes)
        known_set = set(known_changes)

        true_positives = len(detected_set & known_set)
        false_positives = len(detected_set - known_set)
        false_negatives = len(known_set - detected_set)

        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * precision * recall / max(1, precision + recall)

        return precision, recall, f1

    def get_change_statistics(self):
        """Get aggregate statistics across all processed timelines."""
        total_points = len(self.all_change_points) + len(self.all_no_change_points)
        change_rate = len(self.all_change_points) / max(1, total_points)

        return {
            'total_change_points': len(self.all_change_points),
            'total_no_change_points': len(self.all_no_change_points),
            'overall_change_rate': change_rate,
            'detection_parameters': {
                'threshold': self.threshold,
                'drift': self.drift,
                'min_separation': self.min_change_separation
            }
        }