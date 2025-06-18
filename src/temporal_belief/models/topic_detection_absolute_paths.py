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
# For M4 Mac, use MPS if available, otherwise CPU
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
    'climate change and energy policy',  # Changed: more specific
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


def classify_thread_topic(thread_title, confidence_threshold=0.15):  # Lowered from 0.2 to 0.15
    """
    Classify a thread title into political topics using zero-shot classification

    Args:
        thread_title (str): The title of the discussion thread
        confidence_threshold (float): Minimum confidence to assign a topic

    Returns:
        dict: Topic classification results
    """
    if not thread_title or len(thread_title.strip()) < 10:
        return {
            'topic': 'general_discussion',
            'confidence': 0.0,
            'all_scores': {}
        }

    try:
        result = topic_classifier(thread_title, political_topics)

        # If top prediction is below threshold, mark as general
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

    Args:
        corpus_path_or_corpus: Either path to corpus (str) or already-loaded Corpus object
        output_path (str): Path to save annotated corpus (optional)
        max_conversations (int): Maximum number of conversations to process (None = all)
        sample_strategy (str): How to sample conversations ('first', 'random', 'diverse')
    """

    # Load corpus if path provided, otherwise use existing corpus
    if isinstance(corpus_path_or_corpus, str):
        print(f"Loading corpus from {corpus_path_or_corpus}...")
        corpus = convokit.Corpus(filename=corpus_path_or_corpus)
        corpus_path = corpus_path_or_corpus
    else:
        print("Using already-loaded corpus...")
        corpus = corpus_path_or_corpus
        corpus_path = "loaded_corpus"  # fallback for output path naming

    # Create output path if not provided - SAVE TO CURRENT DIRECTORY
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if max_conversations:
            output_path = f"./PoliticalDiscussion_subset_{max_conversations}conv_{timestamp}"
        else:
            output_path = f"./PoliticalDiscussion_full_{timestamp}"

    # If output_path is provided but uses relative paths, convert to current directory
    elif output_path.startswith("../"):
        # Extract just the filename from the path
        filename = os.path.basename(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./{filename}_{timestamp}"

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
            random.seed(42)  # For reproducibility
            conversations_to_process = random.sample(all_conversations, max_conversations)

        elif sample_strategy == 'diverse':
            # Try to get diverse conversations by spreading across the dataset
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

        # Get thread title AND first post content for better classification
        title = conversation.meta.get('title', '')
        first_utterance = conversation.get_chronological_utterance_list()[0]
        first_post = first_utterance.text

        # Combine title and first post (limit to avoid token limits)
        combined_text = f"{title}. {first_post}"[:500]  # First 500 chars

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

        # Get conversation IDs we want to keep
        conversation_ids_to_keep = {conv.id for conv in conversations_to_process}

        # Filter utterances to only include those from selected conversations
        all_utterances = list(corpus.iter_utterances())
        utterances_to_keep = [utt for utt in all_utterances
                              if utt.conversation_id in conversation_ids_to_keep]

        print(f"Keeping {len(utterances_to_keep)} utterances from {len(conversations_to_process)} conversations")

        # Create new corpus components
        from convokit import Corpus, Utterance, Speaker, Conversation

        new_utterances = {}
        new_conversations = {}
        new_speakers = {}

        # Add selected conversations
        for conv in conversations_to_process:
            new_conversations[conv.id] = conv

        # Add utterances from selected conversations and their speakers
        for utt in utterances_to_keep:
            new_utterances[utt.id] = utt
            if utt.speaker.id not in new_speakers:
                new_speakers[utt.speaker.id] = utt.speaker

        # Create subset corpus
        subset_corpus = Corpus(
            utterances=list(new_utterances.values())
        )

        # Step 3: Add topic metadata to utterances in subset corpus
        print("Adding topic metadata to utterances in subset...")
        for utterance in tqdm(subset_corpus.iter_utterances(), desc="Processing utterances"):
            conv_id = utterance.conversation_id

            if conv_id in conversation_topics:
                topic_data = conversation_topics[conv_id]

                # Add topic metadata to each utterance
                utterance.meta['thread_topic'] = topic_data['topic']
                utterance.meta['topic_confidence'] = topic_data['confidence']
                utterance.meta['is_on_topic'] = topic_data['confidence'] > 0.3
            else:
                # Fallback for missing conversations
                utterance.meta['thread_topic'] = 'unknown'
                utterance.meta['topic_confidence'] = 0.0
                utterance.meta['is_on_topic'] = False

        # Save subset corpus
        print(f"Saving subset corpus to {output_path}...")
        try:
            subset_corpus.dump(output_path)
            print("✅ Subset corpus saved successfully!")
        except Exception as e:
            print(f"❌ Error saving subset corpus: {e}")
            # Try alternative save location
            alt_path = f"./corpus_backup_{datetime.now().strftime('%H%M%S')}"
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

                # Add topic metadata to each utterance
                utterance.meta['thread_topic'] = topic_data['topic']
                utterance.meta['topic_confidence'] = topic_data['confidence']
                utterance.meta['is_on_topic'] = topic_data['confidence'] > 0.3
            else:
                # Fallback for missing conversations
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
            # Try alternative save location
            alt_path = f"./corpus_backup_{datetime.now().strftime('%H%M%S')}"
            print(f"Trying alternative path: {alt_path}")
            corpus.dump(alt_path)
            output_path = alt_path
            print(f"✅ Saved to alternative path: {alt_path}")

        final_corpus = corpus

    # Step 4: Save topic statistics
    try:
        stats_file = os.path.join(output_path, "topic_statistics.json")
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

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("✅ Topic statistics saved successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not save statistics: {e}")

    print("Topic detection completed!")
    print(f"Topic distribution: {topic_counts}")
    print(f"Final output location: {output_path}")

    return final_corpus


def preview_topic_classification(corpus_path, num_examples=50):  # Changed from 10 to 50
    """
    Preview topic classification on a few examples before running full processing
    Returns the loaded corpus to avoid reloading
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

        # Combine title and first post like in main function
        combined_text = f"{title}. {first_post}"[:500]

        if not title or title == 'No title':
            display_title = first_post[:100] + "..."
        else:
            display_title = title

        topic_result = classify_thread_topic(combined_text)

        print(f"Title: {display_title}")
        print(f"First Post: {first_post[:200]}{'...' if len(first_post) > 200 else ''}")  # Show first 200 chars of post
        print(f"Predicted Topic: {topic_result['topic']} (confidence: {topic_result['confidence']:.3f})")
        print("-" * 80)

    return corpus  # Return the loaded corpus


# Main execution
if __name__ == "__main__":
    # Set your corpus path here
    CORPUS_PATH = "../../../data/subreddit-PoliticalDiscussion"

    # Option 1: Preview first to check if it's working correctly
    print("=== PREVIEW MODE ===")
    loaded_corpus = preview_topic_classification(CORPUS_PATH, num_examples=50)

    # Ask user if they want to continue
    response = input("\nDoes the topic classification look reasonable? Continue with processing? (y/n): ")

    if response.lower() == 'y':
        # Ask about subset size
        print("\n=== PROCESSING OPTIONS ===")
        print("1. Process small subset (100 conversations) - for testing stance detection")
        print("2. Process medium subset (500 conversations) - for development")
        print("3. Process large subset (2000 conversations) - for validation")
        print("4. Process everything - full dataset")

        choice = input("Choose option (1-4): ")

        # All outputs will be saved to current directory with timestamps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if choice == '1':
            max_conversations = 100
            strategy = 'diverse'  # Get diverse conversations
            OUTPUT_PATH = f"./PoliticalDiscussion_test_100conv_{timestamp}"
        elif choice == '2':
            max_conversations = 500
            strategy = 'diverse'
            OUTPUT_PATH = f"./PoliticalDiscussion_dev_500conv_{timestamp}"
        elif choice == '3':
            max_conversations = 2000
            strategy = 'random'  # Random sample for validation
            OUTPUT_PATH = f"./PoliticalDiscussion_validation_2000conv_{timestamp}"
        elif choice == '4':
            max_conversations = None
            strategy = 'first'  # Doesn't matter since we're processing all
            OUTPUT_PATH = f"./PoliticalDiscussion_full_{timestamp}"
        else:
            print("Invalid choice. Exiting.")
            exit()

        # Option 2: Run topic detection using the already-loaded corpus
        print(f"\n=== PROCESSING {max_conversations or 'ALL'} CONVERSATIONS ===")
        print(f"Saving to current directory: {os.getcwd()}")

        annotated_corpus = add_topic_metadata_to_corpus(
            loaded_corpus,  # Pass the loaded corpus instead of path
            OUTPUT_PATH,
            max_conversations=max_conversations,
            sample_strategy=strategy
        )

        print("Done! Your original corpus is untouched.")
        print(f"Annotated corpus saved in current directory")

        if max_conversations:
            print(f"\nThis is a subset with {max_conversations} conversations.")
            print("Use this for testing your stance detection before processing the full dataset.")
    else:
        print("Stopping here. You can adjust the topics list or classification parameters.")


# Quick verification function
def verify_topic_metadata(corpus_path):
    """
    Quick check to verify the topic metadata was added correctly
    """
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus path does not exist: {corpus_path}")
        return

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


# Convenience functions for common subset sizes
def create_test_subset(corpus_path, conversations=100):
    """Quick function to create a small test subset"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./test_subset_{conversations}conv_{timestamp}"
    return add_topic_metadata_to_corpus(
        corpus_path,
        output_path,
        max_conversations=conversations,
        sample_strategy='diverse'
    )


def create_dev_subset(corpus_path, conversations=500):
    """Quick function to create a development subset"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./dev_subset_{conversations}conv_{timestamp}"
    return add_topic_metadata_to_corpus(
        corpus_path,
        output_path,
        max_conversations=conversations,
        sample_strategy='diverse'
    )