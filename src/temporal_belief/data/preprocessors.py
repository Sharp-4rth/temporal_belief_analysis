import re
import tqdm
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, levene, shapiro
import pandas as pd

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

    def tokenize_and_lower(self, op_text, reply_path_text, stop_words_set):
        op_words = op_text.lower().split()
        reply_words = reply_path_text.lower().split()

        return (op_words, reply_words)

    # This pattern keeps letters, numbers, whitespace, and apostrophes (for contractions)
    def remove_punctuation(self, op_text, reply_path_text):
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

class ExtractFeatures:

    # Feature extraction functions (return features, not scores)
    def get_politeness_features(self, concatenated_path_text):
        """Fast regex-based approximation - good enough for thesis analysis"""
        text_lower = concatenated_path_text.lower()

        return {
            'politeness_gratitude': len(
                re.findall(r'\b(thank|thanks|grateful|appreciate|gratitude)\b', text_lower)),
            'politeness_apologizing': len(
                re.findall(r'\b(sorry|apolog|excuse me|my bad|my mistake)\b', text_lower)),
            'politeness_please': len(re.findall(r'\bplease\b', text_lower)),
            'politeness_indirect_greeting': len(re.findall(r'\b(hello|hi|hey|greetings)\b', text_lower)),
            'politeness_please_start': 1 if re.match(r'^\s*please\b', text_lower) else 0,
            'politeness_hashedge': len(
                re.findall(r'\b(maybe|perhaps|might|could|would|possibly|probably|seems|appears)\b', text_lower)),
            'politeness_deference': len(
                re.findall(r'\b(sir|madam|mr\.|mrs\.|ms\.|dr\.|professor|respectfully)\b', text_lower)),
        }

    def extract_argument_complexity_features(self, text):
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        subordinating = ['because', 'since', 'although', 'while', 'whereas', 'if']

        return {
            'word_count': len(words),
            'unique_words': len(set(words)),
            'sentence_count': len(sentences),
            'subordinating_count': sum(text.lower().count(word) for word in subordinating)
        }

    def extract_evidence_features(self, text):
        import re
        evidence_patterns = [
            r'http[s]?://\S+',
            r'according to',
            r'research shows',
            r'studies indicate',
            r'data suggests',
            r'statistics show',
            r'survey found',
            r'report states'
        ]

        evidence_counts = {}
        for i, pattern in enumerate(evidence_patterns):
            evidence_counts[f'evidence_type_{i}'] = len(re.findall(pattern, text.lower()))

        return evidence_counts

    def extract_hedging_features(self, text):
        hedges = [
            'might', 'could', 'perhaps', 'possibly', 'probably', 'likely',
            'seems', 'appears', 'suggests', 'indicates', 'tends to',
            'generally', 'usually', 'often', 'sometimes', 'may'
        ]

        hedge_counts = {}
        for hedge in hedges:
            hedge_counts[f'hedge_{hedge}'] = text.lower().count(hedge)

        return {
            'hedge_counts': hedge_counts,
            'total_words': len(text.split())
        }

    # Scoring functions (take features, return single score)
    def calculate_complexity_score(self, features):
        if features['word_count'] == 0:
            return 0

        lexical_diversity = features['unique_words'] / features['word_count']
        avg_sentence_length = features['word_count'] / max(1, features['sentence_count'])
        subordinating_ratio = features['subordinating_count'] / features['word_count']

        return lexical_diversity + (avg_sentence_length / 100) + subordinating_ratio

    def calculate_evidence_score(self, features):
        return sum(features.values())

    def calculate_hedging_score_from_features(self, features):
        total_hedges = sum(features['hedge_counts'].values())
        return total_hedges / max(1, features['total_words'])

class GroupPreprocessor:

    def filter_groups(self, groups, groups_tuple):
        # Calculate activity per user in treatment group
        treatment_total = 0
        control_total = 0
        for group_idx, group in enumerate(tqdm(groups_tuple, desc="Processing groups")):
            for user_id, topic_timelines in group.items():
                for topic_timeline in topic_timelines.values():
                    for change_point in topic_timeline.keys():
                        if group_idx == 0:  # Iterate through change points (keys)
                            treatment_total += 1
                        elif group_idx == 1:
                            control_total += 1

        print(f"treatment: {treatment_total}")
        # print(f"Control: {control_total}")

        treatment_activity = []
        for user_id, timelines in groups['with_changes'].items():
            total_points = sum(len(timeline) for timeline in timelines.values())
            treatment_activity.append(total_points)

        target_activity = sum(treatment_activity) // len(treatment_activity)  # Average activity
        target_total = treatment_total  # Match treatment group size

        # Filter control group users by similar activity level
        filtered_control = {}
        control_total = 0

        for user_id, timelines in groups['no_changes'].items():
            user_activity = sum(len(timeline) for timeline in timelines.values())

            # Keep users with similar activity level
            if target_activity * 0.5 <= user_activity <= target_activity * 2:
                filtered_control[user_id] = timelines
                control_total += user_activity

                # Stop when we reach target total
                if control_total >= target_total:
                    break

        # Replace control group
        groups_tuple = (groups['with_changes'], filtered_control)
        print(f"Filtered control group: {len(filtered_control)} users, ~{control_total} total points")

        return groups_tuple

    def get_target(self, groups_tuple):
        # Calculate the number of utterances in each group
        print("Calculating group sizes...")
        group_sizes = []

        for group_idx, group in enumerate(tqdm(groups_tuple, desc="Counting utterances in groups")):
            utterance_count = 0

            for user_id, topic_timelines in group.items():
                for topic_timeline in topic_timelines.values():
                    utterance_count += len(topic_timeline.keys())  # Each key is a change point/utterance

            group_sizes.append(utterance_count)
            print(f"Group {group_idx + 1}: {utterance_count} utterances")

        # Set target_utterances to the smallest group size
        target_utterances = min(group_sizes)
        print(f"\nSmallest group has {target_utterances} utterances")
        print(f"Setting target_utterances = {target_utterances}")

        # Show the sampling strategy
        print(f"\nSampling strategy:")
        for group_idx, size in enumerate(group_sizes):
            print(
                f"Group {group_idx + 1}: {size} total → {target_utterances} sampled ({target_utterances / size * 100:.1f}%)")

        return target_utterances

    def run_statistical_comparison(self, group_scores, alpha=0.05):


        def cohen_d(group1, group2):
            """Calculate Cohen's d for effect size"""
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(
                ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std

        def interpret_effect_size(d):
            """Interpret Cohen's d effect size"""
            abs_d = abs(d)
            if abs_d < 0.2:
                return "Negligible"
            elif abs_d < 0.5:
                return "Small"
            elif abs_d < 0.8:
                return "Medium"
            else:
                return "Large"

        print("=== STATISTICAL SIGNIFICANCE TESTING ===\n")

        if len(group_scores) < 2:
            print("Error: Need at least 2 groups for comparison")
        else:
            # Extract scores for the two groups
            group1_scores = group_scores[0]
            group2_scores = group_scores[1]

            # Results storage
            results_df = []

            print(f"Group 1 sample sizes: {[len(scores) for scores in group1_scores.values()]}")
            print(f"Group 2 sample sizes: {[len(scores) for scores in group2_scores.values()]}")
            print()

            # Test each predictor
            for predictor in group1_scores.keys():
                print(f"=== {predictor.upper()} ===")

                # Get scores for both groups
                g1_scores = np.array(group1_scores[predictor])
                g2_scores = np.array(group2_scores[predictor])

                # Skip if either group has no scores
                if len(g1_scores) == 0 or len(g2_scores) == 0:
                    print(f"Skipping {predictor}: One or both groups have no scores\n")
                    continue

                # Basic descriptive statistics
                g1_mean, g1_std = np.mean(g1_scores), np.std(g1_scores, ddof=1)
                g2_mean, g2_std = np.mean(g2_scores), np.std(g2_scores, ddof=1)

                print(f"Group 1: μ = {g1_mean:.4f}, σ = {g1_std:.4f}, n = {len(g1_scores)}")
                print(f"Group 2: μ = {g2_mean:.4f}, σ = {g2_std:.4f}, n = {len(g2_scores)}")

                # Calculate difference and percentage change
                difference = g1_mean - g2_mean
                percent_change = (difference / g2_mean * 100) if g2_mean != 0 else 0
                print(f"Difference: {difference:.4f} ({percent_change:+.1f}%)")

                # Test for normality (if sample size allows)
                normal_g1 = normal_g2 = None
                if len(g1_scores) >= 3:
                    _, p_norm_g1 = shapiro(g1_scores[:5000] if len(g1_scores) > 5000 else g1_scores)
                    normal_g1 = p_norm_g1 > 0.05
                if len(g2_scores) >= 3:
                    _, p_norm_g2 = shapiro(g2_scores[:5000] if len(g2_scores) > 5000 else g2_scores)
                    normal_g2 = p_norm_g2 > 0.05

                # Test for equal variances
                equal_var = None
                if len(g1_scores) >= 2 and len(g2_scores) >= 2:
                    _, p_levene = levene(g1_scores, g2_scores)
                    equal_var = p_levene > 0.05

                print(f"Normality: G1={normal_g1}, G2={normal_g2}")
                print(f"Equal variances: {equal_var}")

                # Choose appropriate test
                if normal_g1 and normal_g2 and equal_var:
                    # Two-sample t-test (equal variances)
                    t_stat, p_value = stats.ttest_ind(g1_scores, g2_scores, equal_var=True)
                    test_used = "Two-sample t-test (equal var)"
                elif normal_g1 and normal_g2 and not equal_var:
                    # Welch's t-test (unequal variances)
                    t_stat, p_value = stats.ttest_ind(g1_scores, g2_scores, equal_var=False)
                    test_used = "Welch's t-test (unequal var)"
                else:
                    # Mann-Whitney U test (non-parametric)
                    u_stat, p_value = mannwhitneyu(g1_scores, g2_scores, alternative='two-sided')
                    test_used = "Mann-Whitney U test"
                    t_stat = u_stat

                # Effect size (Cohen's d)
                effect_size = cohen_d(g1_scores, g2_scores)
                effect_interpretation = interpret_effect_size(effect_size)

                # Significance interpretation
                if p_value < 0.001:
                    significance = "***"
                    sig_text = "p < 0.001"
                elif p_value < 0.01:
                    significance = "**"
                    sig_text = "p < 0.01"
                elif p_value < 0.05:
                    significance = "*"
                    sig_text = "p < 0.05"
                elif p_value < 0.1:
                    significance = "."
                    sig_text = "p < 0.1 (marginal)"
                else:
                    significance = ""
                    sig_text = "not significant"

                print(f"Test used: {test_used}")
                print(f"Test statistic: {t_stat:.4f}")
                print(f"p-value: {p_value:.6f} {significance}")
                print(f"Result: {sig_text}")
                print(f"Effect size (Cohen's d): {effect_size:.4f} ({effect_interpretation})")

                # Store results
                results_df.append({
                    'Predictor': predictor,
                    'Group_1_Mean': g1_mean,
                    'Group_1_SD': g1_std,
                    'Group_1_N': len(g1_scores),
                    'Group_2_Mean': g2_mean,
                    'Group_2_SD': g2_std,
                    'Group_2_N': len(g2_scores),
                    'Difference': difference,
                    'Percent_Change': percent_change,
                    'Test_Used': test_used,
                    'Test_Statistic': t_stat,
                    'P_Value': p_value,
                    'Significance': significance,
                    'Effect_Size_d': effect_size,
                    'Effect_Interpretation': effect_interpretation,
                    'Significant': p_value < 0.05
                })

                print("-" * 50)
                print()

            # Create summary table
            results_df = pd.DataFrame(results_df)

            print("=== SUMMARY TABLE ===")
            summary_table = results_df[['Predictor', 'Group_1_Mean', 'Group_2_Mean', 'Difference',
                                        'P_Value', 'Significance', 'Effect_Size_d', 'Effect_Interpretation']]
            print(summary_table.to_string(index=False, float_format='%.4f'))

            print(f"\n=== OVERALL RESULTS ===")
            significant_predictors = results_df[results_df['Significant'] == True]
            print(f"Significant predictors (p < 0.05): {len(significant_predictors)}/{len(results_df)}")

            if len(significant_predictors) > 0:
                print("\nSignificant findings:")
                for _, row in significant_predictors.iterrows():
                    direction = "higher" if row['Difference'] > 0 else "lower"
                    print(f"  • {row['Predictor']}: Group 1 {direction} than Group 2")
                    print(f"    Difference: {row['Difference']:.4f} ({row['Percent_Change']:+.1f}%)")
                    print(
                        f"    p = {row['P_Value']:.6f}, d = {row['Effect_Size_d']:.4f} ({row['Effect_Interpretation']})")

            # Multiple comparison correction (Bonferroni)
            n_tests = len(results_df)
            bonferroni_alpha = 0.05 / n_tests
            bonferroni_significant = results_df[results_df['P_Value'] < bonferroni_alpha]

            print(f"\n=== MULTIPLE COMPARISON CORRECTION ===")
            print(f"Bonferroni corrected α = 0.05/{n_tests} = {bonferroni_alpha:.6f}")
            print(f"Significant after correction: {len(bonferroni_significant)}/{len(results_df)}")

            if len(bonferroni_significant) > 0:
                print("\nBonferroni-corrected significant findings:")
                for _, row in bonferroni_significant.iterrows():
                    direction = "higher" if row['Difference'] > 0 else "lower"
                    print(f"  • {row['Predictor']}: Group 1 {direction} than Group 2")
                    print(f"    p = {row['P_Value']:.6f} < {bonferroni_alpha:.6f}")

            # Store results for later use
            globals()['statistical_results'] = results_df
            print(f"\nResults saved to 'statistical_results' DataFrame")

        return results_df