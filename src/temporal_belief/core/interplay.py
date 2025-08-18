import re

class Interplay:

    def calculate_interplay_features(self, op_text, reply_path_text, stop_words_set):
        """Calculate 12 interplay features between OP and reply path."""

        # Remove punctuation
        op_text = re.sub(r"[^\w\s']", '', op_text)
        reply_path_text = re.sub(r"[^\w\s']", '', reply_path_text)

        # Tokenize and clean
        op_words = op_text.lower().split()
        reply_words = reply_path_text.lower().split()

        # Create word sets
        op_all = set(op_words)
        reply_all = set(reply_words)
        op_stop = set(w for w in op_words if w in stop_words_set)
        reply_stop = set(w for w in reply_words if w in stop_words_set)
        op_content = set(w for w in op_words if w not in stop_words_set)
        reply_content = set(w for w in reply_words if w not in stop_words_set)

        # Calculate 4 metrics for each word type
        features = {}

        for word_type, (op_set, reply_set) in [
            ('all', (op_all, reply_all)),
            ('stop', (op_stop, reply_stop)),
            ('content', (op_content, reply_content))
        ]:
            intersection = len(op_set & reply_set)
            union = len(op_set | reply_set)

            features[f'common_words_{word_type}'] = intersection
            features[f'sim_frac_reply_{word_type}'] = intersection / len(reply_set) if reply_set else 0
            features[f'sim_frac_op_{word_type}'] = intersection / len(op_set) if op_set else 0
            features[f'jaccard_{word_type}'] = intersection / union if union else 0

        return features

    def calculate_persuasion_score(self, interplay_features):
        """
        Calculate persuasion score based on Tan et al.'s CMV findings.
        Higher scores indicate higher persuasion likelihood.
        """

        # Extract the key predictive features
        reply_frac_content = interplay_features.get('sim_frac_reply_content', 0)
        jaccard_content = interplay_features.get('jaccard_content', 0)
        op_frac_stop = interplay_features.get('sim_frac_op_stop', 0)
        reply_frac_all = interplay_features.get('sim_frac_reply_all', 0)

        # Apply their findings (↓↓↓↓ means negative correlation, ↑↑↑↑ means positive)
        score = 0

        # Strongest predictor: less content word similarity → more persuasive
        score += (1 - reply_frac_content) * 0.4  # Weight of 0.4 for strongest predictor

        # Less content overlap → more persuasive
        score += (1 - jaccard_content) * 0.3  # Weight of 0.3

        # More stopword similarity → more persuasive
        score += op_frac_stop * 0.2  # Weight of 0.2

        # Less overall similarity → more persuasive
        score += (1 - reply_frac_all) * 0.1  # Weight of 0.1

        return score