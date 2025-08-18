import re

class StancePreprocessor:
    """Preprocess r/PoliticalDiscussion dataset for stance detection."""

    @staticmethod
    def prepare_text(text):
        clean_text = text.strip()
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        return clean_text

    @staticmethod
    def get_contextual_framing_for_topic(topic, text):
        if topic:
            contextual_text = f"In a discussion about {topic}, this comment states: {text}"
        else:
            contextual_text = f"In a political discussion, this comment states: {text}"
        return contextual_text

    @staticmethod
    def mark_quotes(text):
        """Replace ConvoKit quote markers with standard quotation marks."""

        # Split text into lines for processing
        lines = text.split('\n')
        result_lines = []
        in_quote = False

        for line in lines:
            # Check if line starts a quote (begins with &gt;)
            if line.strip().startswith('&gt;'):
                if not in_quote:
                    # Start of new quote - replace &gt; with opening quote
                    cleaned_line = line.replace('&gt;', '"', 1).lstrip()
                    result_lines.append(cleaned_line)
                    in_quote = True
                else:
                    # Continuation of quote - just remove &gt;
                    cleaned_line = line.replace('&gt;', '', 1).lstrip()
                    result_lines.append(cleaned_line)

            # Check if we're ending a quote (empty line or no more &gt; markers)
            elif in_quote and (line.strip() == '' or not line.strip().startswith('&gt;')):
                # End the quote by adding closing quote to previous line
                if result_lines and not result_lines[-1].strip().endswith('"'):
                    result_lines[-1] = result_lines[-1].rstrip() + '"'

                # Add current line if it's not empty
                if line.strip():
                    result_lines.append(line)
                else:
                    result_lines.append(line)  # Keep empty lines

                in_quote = False

            else:
                # Regular line, not in quote
                result_lines.append(line)

        # Handle case where quote goes to end of text
        if in_quote and result_lines and not result_lines[-1].strip().endswith('"'):
            result_lines[-1] = result_lines[-1].rstrip() + '"'

        return '\n'.join(result_lines)


class ChangeDetectorPreprocessor:
    """Filtering timelines for change detection."""

    @staticmethod
    def filter_for_change_detection(timelines, min_posts_per_topic=5, min_topics_per_user=2, min_confidence=0.0):
        """Filter timelines to only include users/topics suitable for change detection"""
        filtered_timelines = {}

        for user_id, user_timeline in timelines.items():
            filtered_user_timeline = {}

            for topic, topic_posts in user_timeline.items():
                # Filter by confidence (if you have access to corpus here)
                reliable_posts = {}
                for utt_id, stance in topic_posts.items():
                    # You'd need to pass corpus or confidence scores here
                    # For now, assume all posts are reliable
                    reliable_posts[utt_id] = stance

                # Check minimum posts per topic
                if len(reliable_posts) >= min_posts_per_topic:
                    filtered_user_timeline[topic] = reliable_posts

            # Check minimum topics per user
            if len(filtered_user_timeline) >= min_topics_per_user:
                filtered_timelines[user_id] = filtered_user_timeline

        return filtered_timelines


class PairPreprocessor:

    def tokenize_quotes(self, utterance_text):
        lines = utterance_text.split('\n')
        processed_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('&gt;') or line.startswith('>'):
                processed_lines.append('[QUOTE]')
            else:
                processed_lines.append(line)

        return '\n'.join(processed_lines)

    def concatenate_path(self, paths):
        concatenated_paths = {}
        for key, utt_list in paths.items():
            path_text = ''
            for utt in utt_list:
                utt_text_quoted = self.tokenize_quotes(utt.text)
                path_text += utt_text_quoted + ' '
            concatenated_paths[key] = path_text.strip()
        return concatenated_paths

    def tokenize_and_lower(op_text, reply_path_text, stop_words_set):
        op_words = op_text.lower().split()
        reply_words = reply_path_text.lower().split()

        return (op_words, reply_words)

    # This pattern keeps letters, numbers, whitespace, and apostrophes (for contractions)
    def remove_punctuation(op_text, reply_path_text):
        op_text = re.sub(r"[^\w\s']", '', op_text)
        reply_path_text = re.sub(r"[^\w\s']", '', reply_path_text)

        return op_text, reply_path_text

    def remove_quotes_from_all(self, op_path_pairs):
        marked_pairs = []
        for op_path_pair in op_path_pairs:
            # Process the OP utterance
            op_text = self.tokenize_quotes(op_path_pair[0].text)

            # Process each utterance path
            processed_paths = []
            for utterances in op_path_pair[1].values():
                path = [self.tokenize_quotes(utt.text) for utt in utterances]
                processed_paths.append(path)

            marked_pairs.append((op_text, processed_paths))

        return marked_pairs

    def concatenate_path_in_pair(self, pair):
        op = pair[0]
        paths = pair[1]

        concatenated_paths = self.concatenate_path(paths)

        return (op, concatenated_paths)

    def concatenate_path_in_all_pairs(self, op_path_pairs):
        # op_path_pairs_quoted = self.remove_quotes_from_all(op_path_pairs)
        preprocessed_pairs = []
        for pair in op_path_pairs:
            pair = self.concatenate_path_in_pair(pair)
            preprocessed_pairs.append(pair)

        return preprocessed_pairs

    def clean_and_tokenize(self, op_text, reply_path_text):
        # Step 1: Remove punctuation
        op_text, reply_path_text = self.remove_punctuation(op_text, reply_path_text)

        # Step 2: Tokenize and lowercase
        op_words, reply_words = self.tokenize_and_lower(op_text, reply_path_text)

        return op_words, reply_words