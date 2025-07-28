

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