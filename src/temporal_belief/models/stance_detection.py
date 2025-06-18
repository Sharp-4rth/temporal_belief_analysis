import convokit
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import os
import json
import torch

# Initialize the zero-shot classifier for stance detection
print("Loading BART model for stance classification...")
# For M4 Mac, use MPS if available, otherwise CPU
device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else -1
stance_classifier = pipeline("zero-shot-classification",
                             model="facebook/bart-large-mnli",
                             device=device)

# Define political stance categories
political_stances = [
    'strongly liberal/progressive',
    'moderately liberal/left-leaning',
    'centrist/moderate',
    'moderately conservative/right-leaning',
    'strongly conservative',
    'libertarian/anti-government',
    'authoritarian/pro-strong-government',
    'neutral/informational'
]

# Alternative simplified version (uncomment if you prefer)
# political_stances = [
#     'liberal/progressive',
#     'conservative',
#     'centrist/moderate',
#     'libertarian',
#     'neutral/informational'
# ]


def classify_political_stance(text, confidence_threshold=0.25):
    """
    Classify a comment's political stance using zero-shot classification

    Args:
        text (str): The comment text to classify
        confidence_threshold (float): Minimum confidence to assign a stance

    Returns:
        dict: Stance classification results
    """
    if not text or len(text.strip()) < 10:
        return {
            'stance': 'neutral/informational',
            'confidence': 0.0,
            'all_scores': {}
        }

    # Clean and truncate text for better classification
    clean_text = text.strip()
    if len(clean_text) > 500:  # Limit for better performance
        clean_text = clean_text[:500] + "..."

    try:
        result = stance_classifier(clean_text, political_stances)

        # If top prediction is below threshold, mark as neutral
        top_confidence = result['scores'][0]
        if top_confidence < confidence_threshold:
            stance = 'neutral/informational'
        else:
            stance = result['labels'][0]

        return {
            'stance': stance,
            'confidence': top_confidence,
            'all_scores': dict(zip(result['labels'], result['scores']))
        }
    except Exception as e:
        print(f"Error classifying stance for text: {str(e)}")
        return {
            'stance': 'error',
            'confidence': 0.0,
            'all_scores': {}
        }


def find_processed_corpus():
    """
    Helper function to find your processed corpus with topics
    """
    import glob
    from pathlib import Path

    # Check ConvoKit directory for processed corpus
    convokit_dir = Path.home() / '.convokit' / 'saved-corpora'
    if convokit_dir.exists():
        # Look for PoliticalDiscussion corpora
        corpus_dirs = list(convokit_dir.glob("PoliticalDiscussion*"))
        if corpus_dirs:
            print("Found processed corpora:")
            for i, corpus_dir in enumerate(corpus_dirs):
                print(f"  {i+1}. {corpus_dir.name}")

            if len(corpus_dirs) == 1:
                return str(corpus_dirs[0])
            else:
                choice = input(f"Choose corpus (1-{len(corpus_dirs)}): ")
                try:
                    idx = int(choice) - 1
                    return str(corpus_dirs[idx])
                except (ValueError, IndexError):
                    print("Invalid choice, using first corpus")
                    return str(corpus_dirs[0])
        else:
            print("No processed PoliticalDiscussion corpora found")
            return None
    else:
        print("ConvoKit directory not found")
        return None


def preview_stance_classification(corpus_path, num_examples=30, topic_filter=None):
    """
    Preview stance classification on sample comments before running full processing

    Args:
        corpus_path (str): Path to corpus
        num_examples (int): Number of examples to show
        topic_filter (str): Optional - only show comments from specific topic
    """
    print("Loading corpus for stance classification preview...")
    corpus = convokit.Corpus(filename=corpus_path)

    print(f"\nPreviewing stance classification on {num_examples} comments:")
    if topic_filter:
        print(f"Filtering for topic: {topic_filter}")
    print("-" * 100)

    utterances = list(corpus.iter_utterances())

    # Filter by topic if specified
    if topic_filter:
        utterances = [u for u in utterances if u.meta.get('thread_topic') == topic_filter]
        print(f"Found {len(utterances)} utterances with topic '{topic_filter}'")

    # Take sample
    sample_utterances = utterances[:num_examples]

    for utt in sample_utterances:
        text = utt.text
        topic = utt.meta.get('thread_topic', 'unknown')

        # Skip very short or empty texts
        if len(text.strip()) < 20:
            continue

        stance_result = classify_political_stance(text)

        print(f"Topic: {topic}")
        print(f"Text: {text[:300]}{'...' if len(text) > 300 else ''}")
        print(f"Predicted Stance: {stance_result['stance']} (confidence: {stance_result['confidence']:.3f})")

        # Show top 3 predictions for context
        if stance_result['all_scores']:
            top_3 = list(stance_result['all_scores'].items())[:3]
            print(f"Top 3: {[(stance, f'{score:.3f}') for stance, score in top_3]}")

        print("-" * 100)


def add_stance_metadata_to_corpus(corpus_path, output_path=None, max_utterances=None,
                                  sample_strategy='first'):
    """
    Load corpus, add stance metadata to utterances, and save to new location

    Args:
        corpus_path (str): Path to original corpus
        output_path (str): Path to save annotated corpus (optional)
        max_utterances (int): Maximum number of utterances to process (None = all)
        sample_strategy (str): How to sample utterances ('first', 'random', 'balanced_topics')
    """

    # Load the original corpus
    print(f"Loading corpus from {corpus_path}...")
    corpus = convokit.Corpus(filename=corpus_path)

    # Create output path if not provided
    if output_path is None:
        # Get just the corpus name without full path
        base_name = os.path.basename(corpus_path.rstrip('/'))
        if max_utterances:
            output_path = f"{base_name}_with_stances_subset_{max_utterances}"
        else:
            output_path = f"{base_name}_with_stances"

    print(f"Will save stance-annotated corpus to: {output_path}")

    # Get all utterances and apply sampling strategy
    all_utterances = list(corpus.iter_utterances())
    total_available = len(all_utterances)

    if max_utterances is None or max_utterances >= total_available:
        utterances_to_process = all_utterances
        print(f"Processing all {total_available} utterances...")
    else:
        print(f"Selecting {max_utterances} utterances from {total_available} total using '{sample_strategy}' strategy...")

        if sample_strategy == 'first':
            utterances_to_process = all_utterances[:max_utterances]

        elif sample_strategy == 'random':
            import random
            random.seed(42)  # For reproducibility
            utterances_to_process = random.sample(all_utterances, max_utterances)

        elif sample_strategy == 'balanced_topics':
            # Get balanced sample across topics
            topic_groups = {}
            for utt in all_utterances:
                topic = utt.meta.get('thread_topic', 'unknown')
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(utt)

            # Calculate utterances per topic
            num_topics = len(topic_groups)
            per_topic = max_utterances // num_topics
            remainder = max_utterances % num_topics

            utterances_to_process = []
            for i, (topic, topic_utterances) in enumerate(topic_groups.items()):
                # Add one extra for first 'remainder' topics
                topic_count = per_topic + (1 if i < remainder else 0)
                topic_sample = topic_utterances[:min(topic_count, len(topic_utterances))]
                utterances_to_process.extend(topic_sample)
                print(f"  {topic}: {len(topic_sample)} utterances")

        else:
            raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

    # Process selected utterances
    print(f"Classifying political stances for {len(utterances_to_process)} utterances...")

    stance_counts = {}
    processed_count = 0

    for utterance in tqdm(utterances_to_process, desc="Processing utterances"):
        text = utterance.text

        # Skip very short texts
        if len(text.strip()) < 10:
            utterance.meta['political_stance'] = 'neutral/informational'
            utterance.meta['stance_confidence'] = 0.0
            utterance.meta['stance_scores'] = {}
            continue

        # Classify stance
        stance_result = classify_political_stance(text)

        # Add metadata to utterance
        utterance.meta['political_stance'] = stance_result['stance']
        utterance.meta['stance_confidence'] = stance_result['confidence']
        utterance.meta['stance_scores'] = stance_result['all_scores']

        # Track statistics
        stance = stance_result['stance']
        stance_counts[stance] = stance_counts.get(stance, 0) + 1
        processed_count += 1

        # Progress update for smaller datasets
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(utterances_to_process)} utterances...")

    # If we're processing a subset, create a new corpus with only those utterances
    if max_utterances and max_utterances < total_available:
        print("Creating subset corpus...")

        # Create new corpus with filtered utterances
        subset_corpus = convokit.Corpus(utterances=utterances_to_process)

        # Save the subset corpus
        print(f"Saving subset corpus to {output_path}...")
        subset_corpus.dump(output_path)
        final_corpus = subset_corpus
    else:
        # Save the full annotated corpus
        print(f"Saving full annotated corpus to {output_path}...")
        corpus.dump(output_path)
        final_corpus = corpus

    # Save stance statistics
    try:
        stats_file = os.path.expanduser(f"~/.convokit/saved-corpora/{output_path}/stance_statistics.json")
        stats = {
            'total_utterances_in_corpus': total_available,
            'processed_utterances': processed_count,
            'sampling_strategy': sample_strategy,
            'max_utterances_requested': max_utterances,
            'stance_distribution': stance_counts,
            'stance_categories': political_stances
        }

        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("✅ Stance statistics saved successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not save statistics: {e}")

    print("Stance classification completed!")
    print(f"Stance distribution: {stance_counts}")
    print(f"Final output location: ~/.convokit/saved-corpora/{output_path}")

    return final_corpus


def analyze_stance_by_topic(corpus_path):
    """
    Analyze stance distribution across different topics
    """
    print("Loading corpus for stance-topic analysis...")
    corpus = convokit.Corpus(filename=corpus_path)

    topic_stance_analysis = {}

    for utterance in corpus.iter_utterances():
        topic = utterance.meta.get('thread_topic', 'unknown')
        stance = utterance.meta.get('political_stance', 'unknown')

        if topic not in topic_stance_analysis:
            topic_stance_analysis[topic] = {}

        topic_stance_analysis[topic][stance] = topic_stance_analysis[topic].get(stance, 0) + 1

    # Print analysis
    print("\n=== STANCE DISTRIBUTION BY TOPIC ===")
    for topic, stance_counts in topic_stance_analysis.items():
        total = sum(stance_counts.values())
        print(f"\n{topic.upper()} ({total} comments):")

        # Sort by count and show percentages
        sorted_stances = sorted(stance_counts.items(), key=lambda x: x[1], reverse=True)
        for stance, count in sorted_stances:
            pct = (count / total) * 100
            print(f"  {stance}: {count} ({pct:.1f}%)")

    return topic_stance_analysis


# Main execution
if __name__ == "__main__":
    # Try to automatically find your processed corpus
    print("=== FINDING PROCESSED CORPUS ===")
    CORPUS_PATH = find_processed_corpus()

    if not CORPUS_PATH:
        # Fallback: manual entry
        print("Could not automatically find processed corpus.")
        CORPUS_PATH = input("Please enter the corpus name (e.g., 'PoliticalDiscussion_subset_100conv_20250618_110426'): ")
        if not CORPUS_PATH.startswith('~') and not os.path.isabs(CORPUS_PATH):
            # Add ConvoKit path if just given name
            CORPUS_PATH = os.path.expanduser(f"~/.convokit/saved-corpora/{CORPUS_PATH}")

    print(f"Using corpus: {CORPUS_PATH}")

    # Check if corpus exists and has topic metadata
    try:
        test_corpus = convokit.Corpus(filename=CORPUS_PATH)
        sample_utt = next(test_corpus.iter_utterances())
        if 'thread_topic' not in sample_utt.meta:
            print("⚠️ Warning: This corpus doesn't seem to have topic metadata!")
            print("Make sure you're using a corpus that has been processed with topic detection.")
        else:
            print("✅ Corpus has topic metadata - ready for stance detection!")
    except Exception as e:
        print(f"❌ Error loading corpus: {e}")
        exit()

    # Option 1: Preview stance classification
    print("\n=== STANCE CLASSIFICATION PREVIEW ===")

    # General preview
    preview_stance_classification(CORPUS_PATH, num_examples=15)

    # Ask user if they want to continue
    response = input("\nDoes the stance classification look reasonable? Continue with processing? (y/n): ")

    if response.lower() == 'y':
        # Ask about subset size
        print("\n=== PROCESSING OPTIONS ===")
        print("0. Process tiny subset (100 utterances) - for quick testing")  # NEW OPTION
        print("1. Process small subset (1000 utterances) - for testing")
        print("2. Process medium subset (5000 utterances) - for development")
        print("3. Process large subset (20000 utterances) - for validation")
        print("4. Process everything - full dataset")

        choice = input("Choose option (0-4): ")

        if choice == '0':  # NEW OPTION
            max_utterances = 100
            strategy = 'balanced_topics'
            print("Creating tiny test set with balanced topics...")
        elif choice == '1':
            max_utterances = 1000
            strategy = 'balanced_topics'  # Get representation from all topics
        elif choice == '2':
            max_utterances = 5000
            strategy = 'balanced_topics'
        elif choice == '3':
            max_utterances = 20000
            strategy = 'random'  # Random sample for validation
        elif choice == '4':
            max_utterances = None
            strategy = 'first'  # Doesn't matter since we're processing all
        else:
            print("Invalid choice. Exiting.")
            exit()

        # Option 2: Run stance detection with chosen parameters
        print(f"\n=== PROCESSING {max_utterances or 'ALL'} UTTERANCES ===")
        annotated_corpus = add_stance_metadata_to_corpus(
            CORPUS_PATH,
            max_utterances=max_utterances,
            sample_strategy=strategy
        )

        # Option 3: Analyze stance distribution by topic
        print("\n=== ANALYZING STANCE-TOPIC RELATIONSHIPS ===")

        # Get the output path for analysis
        base_name = os.path.basename(CORPUS_PATH.rstrip('/'))
        if max_utterances:
            output_name = f"{base_name}_with_stances_subset_{max_utterances}"
        else:
            output_name = f"{base_name}_with_stances"

        output_path = os.path.expanduser(f"~/.convokit/saved-corpora/{output_name}")
        topic_stance_analysis = analyze_stance_by_topic(output_path)

        print("Done! Your corpus now has both topic and stance annotations.")
        print(f"Annotated corpus saved to: ~/.convokit/saved-corpora/{output_name}")

        if max_utterances:
            print(f"\nThis is a subset with {max_utterances} utterances.")
            print("Use this for testing your belief change analysis before processing the full dataset.")
    else:
        print("Stopping here. You can adjust the stance categories or classification parameters.")


# Convenience functions for common subset sizes
def create_tiny_test_subset(corpus_path, size=100):
    """Quick function to create a tiny balanced test subset"""
    return add_stance_metadata_to_corpus(
        corpus_path,
        max_utterances=size,
        sample_strategy='balanced_topics'
    )

def create_test_subset(corpus_path, size=1000):
    """Quick function to create a balanced test subset"""
    return add_stance_metadata_to_corpus(
        corpus_path,
        max_utterances=size,
        sample_strategy='balanced_topics'
    )

def create_dev_subset(corpus_path, size=5000):
    """Quick function to create a development subset"""
    return add_stance_metadata_to_corpus(
        corpus_path,
        max_utterances=size,
        sample_strategy='balanced_topics'
    )


# Quick verification function
def verify_stance_metadata(corpus_path):
    """
    Quick check to verify the stance metadata was added correctly
    """
    corpus = convokit.Corpus(filename=corpus_path)

    print("Checking stance metadata...")
    sample_utterances = list(corpus.iter_utterances())[:5]

    for utt in sample_utterances:
        print(f"Text: {utt.text[:100]}...")
        print(f"Topic: {utt.meta.get('thread_topic', 'NOT FOUND')}")
        print(f"Stance: {utt.meta.get('political_stance', 'NOT FOUND')}")
        print(f"Confidence: {utt.meta.get('stance_confidence', 'NOT FOUND')}")
        print("-" * 80)


# Example usage for testing specific topics:
def test_specific_topics(corpus_path):
    """
    Test stance classification on specific political topics
    """
    interesting_topics = [
        'healthcare policy',
        'gun rights and control',
        'abortion and reproductive rights',
        'climate change and energy policy',
        'immigration policy'
    ]

    for topic in interesting_topics:
        print(f"\n=== TESTING TOPIC: {topic.upper()} ===")
        preview_stance_classification(corpus_path, num_examples=10, topic_filter=topic)