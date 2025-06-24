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

# Define political topics for classification
political_topics = [
    'healthcare policy',
    'immigration policy',
    'economic policy',
    'gun rights and control',
    'abortion and reproductive rights',
    'climate change and energy policy',
    'foreign policy and defense',
    'civil rights and social issues',
    'taxation and government spending',
    'education policy',
    'criminal justice and policing',
    'voting rights and elections',
    'political figures and campaigns',
    'congressional politics',
    'electoral politics',
    'political parties and ideology',
    'media and political commentary'
]


def classify_thread_topic(thread_title, confidence_threshold=0.15):
    """
    Classify a thread title into political topics using zero-shot classification
    """
    if not thread_title or len(thread_title.strip()) < 10:
        return {
            'topic': 'general_discussion',
            'confidence': 0.0,
            'all_scores': {}
        }

    try:
        result = topic_classifier(thread_title, political_topics)
        top_confidence = result['scores'][0]

        if top_confidence < confidence_threshold:
            topic = 'general_discussion'
        else:
            topic = result['labels'][0]

        return {
            'topic': topic,
            'confidence': top_confidence,
            'all_scores': dict(zip(result['labels'], result['scores']))
        }
    except Exception as e:
        print(f"Error classifying '{thread_title}': {e}")
        return {
            'topic': 'error',
            'confidence': 0.0,
            'all_scores': {}
        }


def add_topic_metadata_to_corpus(corpus_path_or_corpus, output_path=None, max_conversations=None,
                                 sample_strategy='first'):
    """
    Load corpus, add topic metadata, and save to new location
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
        if max_conversations:
            output_path = f"PoliticalDiscussion_subset_{max_conversations}conv_{timestamp}"
        else:
            output_path = f"PoliticalDiscussion_full_{timestamp}"

    print(f"Will save annotated corpus to: {output_path}")
    print(f"Current working directory: {os.getcwd()}")

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
    print(f"Classifying thread topics for {len(conversations_to_process)} conversations...")
    conversation_topics = {}

    for conversation in tqdm(conversations_to_process, desc="Processing conversations"):
        conv_id = conversation.id
        title = conversation.meta.get('title', '')
        first_utterance = conversation.get_chronological_utterance_list()[0]
        first_post = first_utterance.text

        # Combine title and first post (limit to avoid token limits)
        combined_text = f"{title}. {first_post}"[:500]
        if not title and len(first_post.strip()) < 20:
            combined_text = "general discussion"

        # Classify the topic using combined text
        topic_result = classify_thread_topic(combined_text)
        conversation_topics[conv_id] = topic_result

        # Add metadata to conversation object
        conversation.meta['detected_topic'] = topic_result['topic']
        conversation.meta['topic_confidence'] = topic_result['confidence']
        conversation.meta['topic_scores'] = topic_result['all_scores']

    # Step 2: If processing subset, create new corpus with only selected conversations
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
                utterance.meta['is_on_topic'] = topic_data['confidence'] > 0.3
            else:
                utterance.meta['thread_topic'] = 'unknown'
                utterance.meta['topic_confidence'] = 0.0
                utterance.meta['is_on_topic'] = False

        # Save subset corpus (this goes to ~/.convokit/saved-corpora/ by default)
        print(f"Saving subset corpus to {output_path}...")
        try:
            subset_corpus.dump(output_path)
            print("✅ Subset corpus saved successfully!")
        except Exception as e:
            print(f"❌ Error saving subset corpus: {e}")
            # Try backup name
            alt_path = f"corpus_backup_{datetime.now().strftime('%H%M%S')}"
            print(f"Trying alternative path: {alt_path}")
            subset_corpus.dump(alt_path)
            output_path = alt_path
            print(f"✅ Saved to alternative path: {alt_path}")

        final_corpus = subset_corpus

    else:
        # Step 3: Propagate topic metadata to all utterances in full corpus
        print("Adding topic metadata to all utterances...")
        utterances = list(corpus.iter_utterances())
        for utterance in tqdm(utterances, desc="Processing utterances"):
            conv_id = utterance.conversation_id
            if conv_id in conversation_topics:
                topic_data = conversation_topics[conv_id]
                utterance.meta['thread_topic'] = topic_data['topic']
                utterance.meta['topic_confidence'] = topic_data['confidence']
                utterance.meta['is_on_topic'] = topic_data['confidence'] > 0.3
            else:
                utterance.meta['thread_topic'] = 'unknown'
                utterance.meta['topic_confidence'] = 0.0
                utterance.meta['is_on_topic'] = False

        # Save full annotated corpus
        print(f"Saving full annotated corpus to {output_path}...")
        try:
            corpus.dump(output_path)
            print("✅ Full corpus saved successfully!")
        except Exception as e:
            print(f"❌ Error saving full corpus: {e}")
            alt_path = f"corpus_backup_{datetime.now().strftime('%H%M%S')}"
            print(f"Trying alternative path: {alt_path}")
            corpus.dump(alt_path)
            output_path = alt_path
            print(f"✅ Saved to alternative path: {alt_path}")

        final_corpus = corpus

    # Step 4: Save topic statistics
    try:
        # This will be saved in the ConvoKit directory alongside the corpus
        stats_file = os.path.expanduser(f"~/.convokit/saved-corpora/{output_path}/topic_statistics.json")
        topic_counts = {}
        for topic_data in conversation_topics.values():
            topic = topic_data['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        stats = {
            'total_conversations_in_original': total_available,
            'processed_conversations': len(conversations_to_process),
            'total_utterances': len(list(final_corpus.iter_utterances())),
            'sampling_strategy': sample_strategy,
            'max_conversations_requested': max_conversations,
            'topic_distribution': topic_counts,
            'topics_detected': list(political_topics) + ['general_discussion', 'unknown', 'error']
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("✅ Topic statistics saved successfully!")
        print(f"Stats saved to: {stats_file}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save statistics: {e}")

    print("Topic detection completed!")
    print(f"Topic distribution: {topic_counts}")
    print(f"Final output location: ~/.convokit/saved-corpora/{output_path}")

    return final_corpus


def preview_topic_classification(corpus_path, num_examples=50):
    """
    Preview topic classification on examples before running full processing
    """
    print("Loading corpus for preview...")
    corpus = convokit.Corpus(filename=corpus_path)

    print(f"\nPreviewing topic classification on {num_examples} conversations:")
    print("-" * 80)

    conversations = list(corpus.iter_conversations())[:num_examples]
    for conv in conversations:
        title = conv.meta.get('title', 'No title')
        first_utterance = conv.get_chronological_utterance_list()[0]
        first_post = first_utterance.text

        combined_text = f"{title}. {first_post}"[:500]
        if not title or title == 'No title':
            display_title = first_post[:100] + "..."
        else:
            display_title = title

        topic_result = classify_thread_topic(combined_text)
        print(f"Title: {display_title}")
        print(f"First Post: {first_post[:200]}{'...' if len(first_post) > 200 else ''}")
        print(f"Predicted Topic: {topic_result['topic']} (confidence: {topic_result['confidence']:.3f})")
        print("-" * 80)

    return corpus


# Main execution
if __name__ == "__main__":
    CORPUS_PATH = "../../../data/subreddit-PoliticalDiscussion"

    print("=== PREVIEW MODE ===")
    loaded_corpus = preview_topic_classification(CORPUS_PATH, num_examples=50)

    response = input("\nDoes the topic classification look reasonable? Continue with processing? (y/n): ")

    if response.lower() == 'y':
        print("\n=== PROCESSING OPTIONS ===")
        print("1. Process small subset (100 conversations) - for testing stance detection")
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

        annotated_corpus = add_topic_metadata_to_corpus(
            loaded_corpus,
            max_conversations=max_conversations,
            sample_strategy=strategy
        )

        print("Done! Your original corpus is untouched.")
        print("Check ~/.convokit/saved-corpora/ for your annotated corpus")

        if max_conversations:
            print(f"\nThis is a subset with {max_conversations} conversations.")
            print("Use this for testing your stance detection before processing the full dataset.")
    else:
        print("Stopping here. You can adjust the topics list or classification parameters.")


def verify_topic_metadata(corpus_path):
    """
    Quick check to verify the topic metadata was added correctly
    """
    try:
        corpus = convokit.Corpus(filename=corpus_path)
        print(f"✅ Corpus loaded successfully from: {corpus_path}")

        print("Checking topic metadata...")
        sample_utterances = list(corpus.iter_utterances())[:5]

        for utt in sample_utterances:
            print(f"Utterance: {utt.text[:50]}...")
            print(f"Topic: {utt.meta.get('thread_topic', 'NOT FOUND')}")
            print(f"Confidence: {utt.meta.get('topic_confidence', 'NOT FOUND')}")
            print("-" * 50)

        # Get topic distribution
        all_utterances = list(corpus.iter_utterances())
        topic_counts = {}
        for utt in all_utterances:
            topic = utt.meta.get('thread_topic', 'unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        print("\n📊 Topic Distribution:")
        for topic, count in topic_counts.items():
            print(f"  - {topic}: {count} utterances")

    except Exception as e:
        print(f"❌ Error verifying corpus: {e}")


def find_corpus_files():
    """
    Helper function to find your saved corpus files
    """
    import glob

    convokit_dir = os.path.expanduser("~/.convokit/saved-corpora/")
    if os.path.exists(convokit_dir):
        corpus_dirs = glob.glob(os.path.join(convokit_dir, "PoliticalDiscussion*"))
        if corpus_dirs:
            print("Found corpus files:")
            for corpus_dir in corpus_dirs:
                print(f"  - {os.path.basename(corpus_dir)}")
                # Check if stats file exists
                stats_file = os.path.join(corpus_dir, "topic_statistics.json")
                if os.path.exists(stats_file):
                    print(f"    (includes topic statistics)")
            return corpus_dirs
        else:
            print("No PoliticalDiscussion corpus files found")
            return []
    else:
        print("ConvoKit directory not found")
        return []


# Convenience functions
def create_test_subset(corpus_path, conversations=100):
    """Quick function to create a small test subset"""
    return add_topic_metadata_to_corpus(
        corpus_path,
        max_conversations=conversations,
        sample_strategy='diverse'
    )


def create_dev_subset(corpus_path, conversations=500):
    """Quick function to create a development subset"""
    return add_topic_metadata_to_corpus(
        corpus_path,
        max_conversations=conversations,
        sample_strategy='diverse'
    )