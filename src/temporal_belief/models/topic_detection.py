import convokit
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import os
import json
import torch

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


def add_topic_metadata_to_corpus(corpus_path, output_path=None):
    """
    Load corpus, add topic metadata, and save to new location

    Args:
        corpus_path (str): Path to original corpus
        output_path (str): Path to save annotated corpus (optional)
    """

    # Load the original corpus
    print(f"Loading corpus from {corpus_path}...")
    corpus = convokit.Corpus(filename=corpus_path)

    # Create output path if not provided
    if output_path is None:
        base_path = corpus_path.rstrip('/')
        output_path = f"{base_path}_with_topics"

    print(f"Will save annotated corpus to: {output_path}")

    # Step 1: Extract all conversation titles and classify topics
    print("Classifying thread topics...")
    conversation_topics = {}

    conversations = list(corpus.iter_conversations())
    for conversation in tqdm(conversations, desc="Processing conversations"):
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

    # Step 2: Propagate topic metadata to all utterances in each conversation
    print("Adding topic metadata to utterances...")

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

    # Step 3: Save the annotated corpus
    print(f"Saving annotated corpus to {output_path}...")
    corpus.dump(output_path)

    # Step 4: Save topic statistics
    stats_file = os.path.join(output_path, "topic_statistics.json")
    topic_counts = {}
    for topic_data in conversation_topics.values():
        topic = topic_data['topic']
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    stats = {
        'total_conversations': len(conversations),
        'total_utterances': len(utterances),
        'topic_distribution': topic_counts,
        'topics_detected': list(political_topics) + ['general_discussion', 'unknown', 'error']
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("Topic detection completed!")
    print(f"Topic distribution: {topic_counts}")

    return corpus


def preview_topic_classification(corpus_path, num_examples=50):  # Changed from 10 to 50
    """
    Preview topic classification on a few examples before running full processing
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


# Main execution
if __name__ == "__main__":
    # Set your corpus path here
    CORPUS_PATH = "../../../data/subreddit-PoliticalDiscussion"
    OUTPUT_PATH = "../../../data/subreddit-PoliticalDiscussion_with_topics"

    # Option 1: Preview first to check if it's working correctly
    print("=== PREVIEW MODE ===")
    preview_topic_classification(CORPUS_PATH, num_examples=50)  # Changed from 20 to 50

    # Ask user if they want to continue
    response = input("\nDoes the topic classification look reasonable? Continue with full processing? (y/n): ")

    if response.lower() == 'y':
        # Option 2: Run full topic detection
        print("\n=== FULL PROCESSING ===")
        annotated_corpus = add_topic_metadata_to_corpus(CORPUS_PATH, OUTPUT_PATH)
        print("Done! Your original corpus is untouched.")
        print(f"Annotated corpus saved to: {OUTPUT_PATH}")
    else:
        print("Stopping here. You can adjust the topics list or classification parameters.")


# Quick verification function
def verify_topic_metadata(corpus_path):
    """
    Quick check to verify the topic metadata was added correctly
    """
    corpus = convokit.Corpus(filename=corpus_path)

    print("Checking topic metadata...")
    sample_utterances = list(corpus.iter_utterances())[:5]

    for utt in sample_utterances:
        print(f"Utterance: {utt.text[:50]}...")
        print(f"Topic: {utt.meta.get('thread_topic', 'NOT FOUND')}")
        print(f"Confidence: {utt.meta.get('topic_confidence', 'NOT FOUND')}")
        print("-" * 50)