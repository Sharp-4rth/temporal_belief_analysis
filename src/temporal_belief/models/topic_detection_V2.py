import convokit
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import os
import json
import torch
from datetime import datetime

# Initialize the zero-shot classifier
print("Loading BART model for topic classification...")
device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else -1
topic_classifier = pipeline("zero-shot-classification",
                            model="facebook/bart-large-mnli",
                            device=device)

# IMPROVED POLITICAL TOPICS - More specific and comprehensive
# Removed vague "general discussion" and added more specific categories
IMPROVED_POLITICAL_TOPICS = [
    # Core Policy Areas
    'healthcare policy and insurance reform',
    'immigration policy and border security',
    'economic policy and fiscal management',
    'gun rights and firearm regulations',
    'abortion and reproductive healthcare rights',
    'climate change and environmental policy',
    'foreign policy and international relations',
    'civil rights and social justice issues',
    'taxation policy and government spending',
    'education policy and school reform',
    'criminal justice reform and policing',
    'voting rights and election integrity',

    # Political Process & Figures
    'political campaigns and candidate analysis',
    'congressional politics and legislation',
    'electoral politics and voting behavior',
    'political parties and ideological movements',
    'media coverage and political commentary',
    'government oversight and institutional reform',

    # Social & Cultural Issues
    'religious freedom and church-state separation',
    'LGBTQ+ rights and gender equality',
    'racial justice and discrimination issues',
    'income inequality and social welfare',
    'drug policy and criminal law reform',
    'technology regulation and privacy rights',
    'trade policy and economic globalization',
    'infrastructure and public works policy',

    # Specialized Areas
    'military and veterans affairs',
    'housing policy and urban development',
    'energy policy and resource management',
    'transportation and public transit policy'
]

# HIERARCHICAL TOPIC CLASSIFICATION
# First classify into broad categories, then specific subcategories
BROAD_CATEGORIES = [
    'domestic policy issues',
    'foreign policy and international affairs',
    'electoral politics and campaigns',
    'social and cultural issues',
    'economic and fiscal policy',
    'constitutional and legal issues',
    'government process and reform',
    'non-political discussion'
]

SUBCATEGORY_MAPPING = {
    'domestic policy issues': [
        'healthcare policy and insurance reform',
        'education policy and school reform',
        'criminal justice reform and policing',
        'gun rights and firearm regulations',
        'immigration policy and border security',
        'climate change and environmental policy',
        'drug policy and criminal law reform',
        'infrastructure and public works policy',
        'housing policy and urban development',
        'energy policy and resource management',
        'transportation and public transit policy'
    ],
    'social and cultural issues': [
        'abortion and reproductive healthcare rights',
        'civil rights and social justice issues',
        'religious freedom and church-state separation',
        'LGBTQ+ rights and gender equality',
        'racial justice and discrimination issues',
        'income inequality and social welfare',
        'technology regulation and privacy rights'
    ],
    'economic and fiscal policy': [
        'taxation policy and government spending',
        'trade policy and economic globalization',
        'economic policy and fiscal management'
    ],
    'electoral politics and campaigns': [
        'political campaigns and candidate analysis',
        'voting rights and election integrity',
        'electoral politics and voting behavior'
    ],
    'constitutional and legal issues': [
        'voting rights and election integrity',
        'civil rights and social justice issues',
        'religious freedom and church-state separation'
    ],
    'government process and reform': [
        'congressional politics and legislation',
        'government oversight and institutional reform',
        'political parties and ideological movements',
        'media coverage and political commentary'
    ],
    'foreign policy and international affairs': [
        'foreign policy and international relations',
        'military and veterans affairs',
        'trade policy and economic globalization'
    ]
}


def classify_thread_topic_improved(thread_title, first_post="", confidence_threshold=0.20):
    """
    IMPROVED topic classification using hierarchical approach and better categories
    1. First classify into broad category
    2. Then classify into specific subcategory within that broad category
    3. Use both title and first post content for better accuracy
    """
    if not thread_title or len(thread_title.strip()) < 10:
        if not first_post or len(first_post.strip()) < 20:
            return {
                'topic': 'non_political_discussion',
                'broad_category': 'non-political discussion',
                'confidence': 0.0,
                'all_scores': {},
                'method': 'insufficient_content'
            }

    # Combine title and first post for richer context
    combined_text = f"{thread_title}. {first_post}"[:800]  # Increased limit for better context

    # Clean up the text
    if not thread_title.strip():
        combined_text = first_post[:600]
    elif not first_post.strip():
        combined_text = thread_title

    try:
        # Step 1: Classify into broad category
        broad_result = topic_classifier(combined_text, BROAD_CATEGORIES)
        broad_category = broad_result['labels'][0]
        broad_confidence = broad_result['scores'][0]

        # If it's clearly non-political, return early
        if broad_category == 'non-political discussion' and broad_confidence > 0.6:
            return {
                'topic': 'non_political_discussion',
                'broad_category': broad_category,
                'confidence': broad_confidence,
                'all_scores': dict(zip(broad_result['labels'], broad_result['scores'])),
                'method': 'hierarchical_early_exit'
            }

        # Step 2: If political, classify into specific subcategory
        if broad_category in SUBCATEGORY_MAPPING and broad_confidence > confidence_threshold:
            relevant_subcategories = SUBCATEGORY_MAPPING[broad_category]

            # Classify into specific subcategory
            specific_result = topic_classifier(combined_text, relevant_subcategories)
            specific_topic = specific_result['labels'][0]
            specific_confidence = specific_result['scores'][0]

            # Use weighted confidence (broad category confidence * specific confidence)
            final_confidence = broad_confidence * specific_confidence

            # If specific classification is confident enough, use it
            if specific_confidence > confidence_threshold:
                final_topic = specific_topic.replace(' ', '_').replace('-', '_')
            else:
                # Fall back to broad category
                final_topic = broad_category.replace(' ', '_').replace('-', '_')
                final_confidence = broad_confidence
        else:
            # Low confidence in broad category, classify against all topics
            all_result = topic_classifier(combined_text, IMPROVED_POLITICAL_TOPICS)
            if all_result['scores'][0] > confidence_threshold:
                final_topic = all_result['labels'][0].replace(' ', '_').replace('-', '_')
                final_confidence = all_result['scores'][0]
                broad_category = 'mixed_classification'
            else:
                final_topic = 'non_political_discussion'
                final_confidence = 1.0 - all_result['scores'][0]
                broad_category = 'non-political discussion'

        return {
            'topic': final_topic,
            'broad_category': broad_category,
            'confidence': final_confidence,
            'all_scores': dict(zip(broad_result['labels'], broad_result['scores'])),
            'method': 'hierarchical_classification'
        }

    except Exception as e:
        print(f"Error classifying '{thread_title}': {e}")
        return {
            'topic': 'classification_error',
            'broad_category': 'error',
            'confidence': 0.0,
            'all_scores': {},
            'method': 'error'
        }


def classify_thread_topic_single_pass(thread_title, first_post="", confidence_threshold=0.20):
    """
    Alternative: Single-pass classification against all improved topics
    """
    if not thread_title or len(thread_title.strip()) < 10:
        if not first_post or len(first_post.strip()) < 20:
            return {
                'topic': 'non_political_discussion',
                'confidence': 0.0,
                'all_scores': {},
                'method': 'single_pass_insufficient'
            }

    # Combine and clean text
    combined_text = f"{thread_title}. {first_post}"[:800]
    if not thread_title.strip():
        combined_text = first_post[:600]
    elif not first_post.strip():
        combined_text = thread_title

    try:
        # Add non-political option to the classification
        all_topics = IMPROVED_POLITICAL_TOPICS + ['non_political_discussion']

        result = topic_classifier(combined_text, all_topics)
        top_confidence = result['scores'][0]
        top_topic = result['labels'][0]

        if top_confidence < confidence_threshold:
            final_topic = 'non_political_discussion'
            final_confidence = 1.0 - top_confidence
        else:
            final_topic = top_topic.replace(' ', '_').replace('-', '_')
            final_confidence = top_confidence

        return {
            'topic': final_topic,
            'confidence': final_confidence,
            'all_scores': dict(zip(result['labels'], result['scores'])),
            'method': 'single_pass_comprehensive'
        }

    except Exception as e:
        print(f"Error in single-pass classification: {e}")
        return {
            'topic': 'classification_error',
            'confidence': 0.0,
            'all_scores': {},
            'method': 'single_pass_error'
        }


def add_topic_metadata_to_corpus_improved(corpus_path_or_corpus, output_path=None, max_conversations=None,
                                          sample_strategy='first', classification_method='hierarchical'):
    """
    IMPROVED topic classification with better categories and methods

    Args:
        classification_method: 'hierarchical' or 'single_pass'
    """
    # Load corpus if path provided, otherwise use existing corpus
    if isinstance(corpus_path_or_corpus, str):
        print(f"Loading corpus from {corpus_path_or_corpus}...")
        corpus = convokit.Corpus(filename=corpus_path_or_corpus)
    else:
        print("Using already-loaded corpus...")
        corpus = corpus_path_or_corpus

    # Create output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_suffix = f"_{classification_method}" if classification_method != 'hierarchical' else "_improved"
        if max_conversations:
            output_path = f"PoliticalDiscussion_subset_{max_conversations}conv{method_suffix}_{timestamp}"
        else:
            output_path = f"PoliticalDiscussion_full{method_suffix}_{timestamp}"

    print(f"Will save annotated corpus to: {output_path}")
    print(f"Using classification method: {classification_method}")

    # Get all conversations and apply sampling strategy
    all_conversations = list(corpus.iter_conversations())
    total_available = len(all_conversations)

    if max_conversations is None or max_conversations >= total_available:
        conversations_to_process = all_conversations
        print(f"Processing all {total_available} conversations...")
    else:
        print(
            f"Selecting {max_conversations} conversations from {total_available} total using '{sample_strategy}' strategy...")

        if sample_strategy == 'first':
            conversations_to_process = all_conversations[:max_conversations]
        elif sample_strategy == 'random':
            import random
            random.seed(42)
            conversations_to_process = random.sample(all_conversations, max_conversations)
        elif sample_strategy == 'diverse':
            step = len(all_conversations) // max_conversations
            conversations_to_process = all_conversations[::step][:max_conversations]
            print(f"  Selected every {step}th conversation for diversity")
        else:
            raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

    # Step 1: Extract selected conversation titles and classify topics
    print(
        f"Classifying thread topics for {len(conversations_to_process)} conversations using {classification_method}...")
    conversation_topics = {}
    classification_stats = {
        'political_conversations': 0,
        'non_political_conversations': 0,
        'high_confidence': 0,
        'low_confidence': 0,
        'broad_category_distribution': {},
        'method_used': classification_method
    }

    for conversation in tqdm(conversations_to_process, desc="Processing conversations"):
        conv_id = conversation.id
        title = conversation.meta.get('title', '')
        first_utterance = conversation.get_chronological_utterance_list()[0]
        first_post = first_utterance.text

        # Use improved classification method
        if classification_method == 'hierarchical':
            topic_result = classify_thread_topic_improved(title, first_post)
        else:  # single_pass
            topic_result = classify_thread_topic_single_pass(title, first_post)

        conversation_topics[conv_id] = topic_result

        # Add metadata to conversation object
        conversation.meta['detected_topic'] = topic_result['topic']
        conversation.meta['topic_confidence'] = topic_result['confidence']
        conversation.meta['topic_method'] = topic_result['method']

        if 'broad_category' in topic_result:
            conversation.meta['broad_category'] = topic_result['broad_category']
            # Track broad category distribution
            broad_cat = topic_result['broad_category']
            classification_stats['broad_category_distribution'][broad_cat] = \
                classification_stats['broad_category_distribution'].get(broad_cat, 0) + 1

        # Track statistics
        if topic_result['topic'] != 'non_political_discussion':
            classification_stats['political_conversations'] += 1
        else:
            classification_stats['non_political_conversations'] += 1

        if topic_result['confidence'] > 0.5:
            classification_stats['high_confidence'] += 1
        else:
            classification_stats['low_confidence'] += 1

    # Step 2: Create corpus and add metadata to utterances
    if max_conversations and max_conversations < total_available:
        print("Creating subset corpus with selected conversations...")

        conversation_ids_to_keep = {conv.id for conv in conversations_to_process}
        all_utterances = list(corpus.iter_utterances())
        utterances_to_keep = [utt for utt in all_utterances
                              if utt.conversation_id in conversation_ids_to_keep]

        print(f"Keeping {len(utterances_to_keep)} utterances from {len(conversations_to_process)} conversations")

        # Create subset corpus
        subset_corpus = convokit.Corpus(utterances=utterances_to_keep)

        # Add topic metadata to utterances in subset corpus
        print("Adding topic metadata to utterances in subset...")
        for utterance in tqdm(subset_corpus.iter_utterances(), desc="Processing utterances"):
            conv_id = utterance.conversation_id
            if conv_id in conversation_topics:
                topic_data = conversation_topics[conv_id]
                utterance.meta['thread_topic'] = topic_data['topic']
                utterance.meta['topic_confidence'] = topic_data['confidence']
                utterance.meta['topic_method'] = topic_data['method']
                utterance.meta['is_political'] = topic_data['topic'] != 'non_political_discussion'
                utterance.meta['is_high_confidence'] = topic_data['confidence'] > 0.5

                if 'broad_category' in topic_data:
                    utterance.meta['broad_category'] = topic_data['broad_category']
            else:
                utterance.meta['thread_topic'] = 'unknown'
                utterance.meta['topic_confidence'] = 0.0
                utterance.meta['is_political'] = False
                utterance.meta['is_high_confidence'] = False

        final_corpus = subset_corpus
    else:
        # Process full corpus
        print("Adding topic metadata to all utterances...")
        utterances = list(corpus.iter_utterances())
        for utterance in tqdm(utterances, desc="Processing utterances"):
            conv_id = utterance.conversation_id
            if conv_id in conversation_topics:
                topic_data = conversation_topics[conv_id]
                utterance.meta['thread_topic'] = topic_data['topic']
                utterance.meta['topic_confidence'] = topic_data['confidence']
                utterance.meta['topic_method'] = topic_data['method']
                utterance.meta['is_political'] = topic_data['topic'] != 'non_political_discussion'
                utterance.meta['is_high_confidence'] = topic_data['confidence'] > 0.5

                if 'broad_category' in topic_data:
                    utterance.meta['broad_category'] = topic_data['broad_category']
            else:
                utterance.meta['thread_topic'] = 'unknown'
                utterance.meta['topic_confidence'] = 0.0
                utterance.meta['is_political'] = False
                utterance.meta['is_high_confidence'] = False

        final_corpus = corpus

    # Save corpus
    print(f"Saving annotated corpus to {output_path}...")
    try:
        final_corpus.dump(output_path)
        print("✅ Corpus saved successfully!")
    except Exception as e:
        print(f"❌ Error saving corpus: {e}")
        alt_path = f"corpus_backup_{datetime.now().strftime('%H%M%S')}"
        print(f"Trying alternative path: {alt_path}")
        final_corpus.dump(alt_path)
        output_path = alt_path
        print(f"✅ Saved to alternative path: {alt_path}")

    # Step 3: Save enhanced statistics
    try:
        stats_file = os.path.expanduser(f"~/.convokit/saved-corpora/{output_path}/topic_statistics_improved.json")
        topic_counts = {}
        for topic_data in conversation_topics.values():
            topic = topic_data['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Calculate political vs non-political ratio
        total_processed = len(conversations_to_process)
        political_ratio = classification_stats[
                              'political_conversations'] / total_processed if total_processed > 0 else 0
        confidence_ratio = classification_stats['high_confidence'] / total_processed if total_processed > 0 else 0

        stats = {
            'total_conversations_in_original': total_available,
            'processed_conversations': len(conversations_to_process),
            'total_utterances': len(list(final_corpus.iter_utterances())),
            'sampling_strategy': sample_strategy,
            'classification_method': classification_method,
            'max_conversations_requested': max_conversations,
            'topic_distribution': topic_counts,
            'classification_performance': {
                'political_conversations': classification_stats['political_conversations'],
                'non_political_conversations': classification_stats['non_political_conversations'],
                'political_ratio': political_ratio,
                'high_confidence_classifications': classification_stats['high_confidence'],
                'low_confidence_classifications': classification_stats['low_confidence'],
                'confidence_ratio': confidence_ratio,
                'broad_category_distribution': classification_stats['broad_category_distribution']
            },
            'available_topics': IMPROVED_POLITICAL_TOPICS + ['non_political_discussion'],
            'broad_categories': BROAD_CATEGORIES
        }

        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("✅ Enhanced topic statistics saved successfully!")
        print(f"Stats saved to: {stats_file}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save statistics: {e}")

    print("Improved topic classification completed!")
    print(f"Topic distribution: {topic_counts}")
    print(f"Political vs Non-Political: {political_ratio:.1%} political")
    print(f"High confidence rate: {confidence_ratio:.1%}")
    print(f"Final output location: ~/.convokit/saved-corpora/{output_path}")

    return final_corpus


def preview_topic_classification_improved(corpus_path, num_examples=30):
    """
    Compare old vs new topic classification methods
    """
    print("Loading corpus for improved topic classification preview...")
    corpus = convokit.Corpus(filename=corpus_path)

    print(f"\nComparing topic classification methods on {num_examples} conversations:")
    print("-" * 120)

    conversations = list(corpus.iter_conversations())[:num_examples]
    for conv in conversations:
        title = conv.meta.get('title', 'No title')
        first_utterance = conv.get_chronological_utterance_list()[0]
        first_post = first_utterance.text

        if not title or title == 'No title':
            display_title = first_post[:80] + "..."
        else:
            display_title = title

        # Test both methods
        hierarchical_result = classify_thread_topic_improved(title, first_post)
        single_pass_result = classify_thread_topic_single_pass(title, first_post)

        print(f"Title: {display_title}")
        print(f"First Post: {first_post[:150]}{'...' if len(first_post) > 150 else ''}")
        print(
            f"Hierarchical:  {hierarchical_result['topic']:<40} (conf: {hierarchical_result['confidence']:.3f}) [{hierarchical_result.get('broad_category', 'N/A')}]")
        print(f"Single-Pass:   {single_pass_result['topic']:<40} (conf: {single_pass_result['confidence']:.3f})")

        # Highlight disagreements
        if hierarchical_result['topic'] != single_pass_result['topic']:
            print("⚠️  DISAGREEMENT between methods!")

        print("-" * 120)

    return corpus


# Main execution
if __name__ == "__main__":
    CORPUS_PATH = "../../../data/subreddit-PoliticalDiscussion"

    print("=== IMPROVED TOPIC CLASSIFICATION PREVIEW ===")
    loaded_corpus = preview_topic_classification_improved(CORPUS_PATH, num_examples=20)

    print("\n=== METHOD COMPARISON ===")
    print("1. Hierarchical: First classify broad category, then specific topic")
    print("2. Single-pass: Classify against all improved topics at once")

    method_choice = input("Choose classification method (1-2): ")
    method_map = {'1': 'hierarchical', '2': 'single_pass'}
    selected_method = method_map.get(method_choice, 'hierarchical')

    response = input(f"\nProceed with {selected_method} method? (y/n): ")

    if response.lower() == 'y':
        print("\n=== PROCESSING OPTIONS ===")
        print("1. Process small subset (100 conversations) - for testing")
        print("2. Process medium subset (500 conversations) - for development")
        print("3. Process large subset (2000 conversations) - for validation")
        print("4. Process everything - full dataset")

        choice = input("Choose option (1-4): ")

        if choice == '1':
            max_conversations = 100
            strategy = 'diverse'
        elif choice == '2':
            max_conversations = 500
            strategy = 'diverse'
        elif choice == '3':
            max_conversations = 2000
            strategy = 'random'
        elif choice == '4':
            max_conversations = None
            strategy = 'first'
        else:
            print("Invalid choice. Exiting.")
            exit()

        print(f"\n=== PROCESSING {max_conversations or 'ALL'} CONVERSATIONS ===")
        print("Files will be saved to ~/.convokit/saved-corpora/")

        annotated_corpus = add_topic_metadata_to_corpus_improved(
            loaded_corpus,
            max_conversations=max_conversations,
            sample_strategy=strategy,
            classification_method=selected_method
        )

        print("Done! Check the improved topic classification results.")

    else:
        print("Stopping here. You can adjust the topics or methods.")