
def preview_topic_classification(corpus_path, num_examples=50):
    """
    Preview topic classification on examples before running full processing
    """
    print("Loading corpus for preview...")
    corpus = convokit.Corpus(filename=corpus_path)

    print(f"\nPreviewing topic classification on {num_examples} conversations:")
    print("-" * 80)

    conversations = list(corpus.iter_conversations())[:num_examples]
    preview_results = []

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

        preview_data = {
            'title': display_title,
            'first_post': first_post[:200] + ('...' if len(first_post) > 200 else ''),
            'topic': topic_result['topic'],
            'confidence': topic_result['confidence']
        }
        preview_results.append(preview_data)

        print(f"Title: {display_title}")
        print(f"First Post: {first_post[:200]}{'...' if len(first_post) > 200 else ''}")
        print(f"Predicted Topic: {topic_result['topic']} (confidence: {topic_result['confidence']:.3f})")
        print("-" * 80)

    # Save preview to Drive/local directory
    preview_file = os.path.join(DRIVE_PROJECT_DIR, f"topic_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(preview_file, 'w') as f:
        json.dump(preview_results, f, indent=2)
    print(f"📄 Preview results saved to: {preview_file}")

    return corpus


# Main execution
if __name__ == "__main__":
    if IN_COLAB:
        CORPUS_PATH = "/content/drive/MyDrive/temporal-belief-analysis/subreddit-PoliticalDiscussion"

        # Check if corpus exists in Drive, if not provide instructions
        if not os.path.exists(CORPUS_PATH):
            print(f"❌ Corpus not found at: {CORPUS_PATH}")
            print("\n📥 Please upload your ConvoKit corpus to Google Drive:")
            print(f"   1. Upload the corpus folder to: {DRIVE_PROJECT_DIR}")
            print("   2. Make sure it's named 'subreddit-PoliticalDiscussion'")
            print("   3. Restart this script")
            exit()
    else:
        # Running locally - adjust path as needed
        CORPUS_PATH = "./subreddit-PoliticalDiscussion"
        if not os.path.exists(CORPUS_PATH):
            print(f"❌ Corpus not found at: {CORPUS_PATH}")
            print("Please adjust CORPUS_PATH to point to your ConvoKit corpus")
            exit()

    print("=== PREVIEW MODE ===")
    loaded_corpus = preview_topic_classification(CORPUS_PATH, num_examples=20)

    print("\n🤖 Does the topic classification look reasonable?")
    response = input("Continue with processing? (y/n): ")

    if response.lower() == 'y':
        print("\n=== PROCESSING OPTIONS ===")
        print("1. 🧪 Test subset (100 conversations) - for testing stance detection")
        print("2. 🔧 Dev subset (500 conversations) - for development")
        print("3. 📊 Large subset (2000 conversations) - for validation")
        print("4. 🌍 Full dataset - process everything")

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
        if IN_COLAB:
            print("📁 Files will be saved to Google Drive and auto-downloaded")
        else:
            print(f"📁 Files will be saved to {DRIVE_PROJECT_DIR}")
        print("🔄 Processing started...")

        annotated_corpus = add_topic_metadata_to_corpus(
            loaded_corpus,
            max_conversations=max_conversations,
            sample_strategy=strategy,
            save_to_drive=True
        )

        print("\n🎉 PROCESSING COMPLETE!")
        print("✅ Corpus annotated with topic metadata")
        if IN_COLAB:
            print("💾 Files saved to Google Drive")
            print("⬇️ Package downloaded to your local system")
        else:
            print(f"💾 Files saved to {DRIVE_PROJECT_DIR}")
        print(f"📂 Storage location: {DRIVE_PROJECT_DIR}")

        # Summary
        print(f"\n📋 SUMMARY:")
        if max_conversations:
            print(f"   - Processed: {max_conversations} conversations (subset)")
        else:
            print(f"   - Processed: All conversations (full dataset)")
        print(f"   - Strategy: {strategy}")
        print(f"   - Files ready for stance detection training!")

    else:
        print("🛑 Stopping here. You can adjust the topics list or classification parameters.")


def quick_test_subset():
    """
    Quick function to create a small test subset for development
    """
    if IN_COLAB:
        CORPUS_PATH = "/content/drive/MyDrive/temporal-belief-analysis/subreddit-PoliticalDiscussion"
    else:
        CORPUS_PATH = "./subreddit-PoliticalDiscussion"

    if not os.path.exists(CORPUS_PATH):
        print(f"❌ Corpus not found at: {CORPUS_PATH}")
        return None

    print("🚀 Creating quick test subset (100 conversations)...")
    corpus = convokit.Corpus(filename=CORPUS_PATH)

    return add_topic_metadata_to_corpus(
        corpus,
        max_conversations=100,
        sample_strategy='diverse',
        save_to_drive=True
    )


def verify_downloaded_corpus(zip_path):
    """
    Verify that the downloaded corpus is working correctly
    """
    import zipfile
    import tempfile

    print(f"🔍 Verifying downloaded corpus: {zip_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find corpus directory
        corpus_dir = os.path.join(temp_dir, "corpus_data")
        if not os.path.exists(corpus_dir):
            print("❌ Corpus data not found in zip file")
            return False

        try:
            # Load corpus
            corpus = convokit.Corpus(filename=corpus_dir)

            # Check utterances
            utterances = list(corpus.iter_utterances())[:5]
            print(f"✅ Corpus loaded successfully: {len(utterances)} utterances checked")

            # Check topic metadata
            for utt in utterances:
                topic = utt.meta.get('thread_topic', 'NOT FOUND')
                confidence = utt.meta.get('topic_confidence', 'NOT FOUND')
                print(f"   - Topic: {topic}, Confidence: {confidence}")

            return True

        except Exception as e:
            print(f"❌ Error loading corpus: {e}")
            return False


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
    import convokit


from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import os
import json
import torch
from datetime import datetime
import zipfile
import shutil

# Check if running in Colab
try:
    from google.colab import drive, files

    IN_COLAB = True
    # Mount Google Drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')

    # Create project directory in Google Drive
    DRIVE_PROJECT_DIR = "/content/drive/MyDrive/temporal-belief-analysis"
    os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
    print(f"Project directory created: {DRIVE_PROJECT_DIR}")
except ImportError:
    IN_COLAB = False
    DRIVE_PROJECT_DIR = "./output"
    os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
    print(f"Running locally. Output directory: {DRIVE_PROJECT_DIR}")

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


def create_downloadable_package(corpus_output_path, topic_stats, max_conversations):
    """
    Create a downloadable package with corpus and metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"political_corpus_{max_conversations or 'full'}_{timestamp}"

    # Create package directory
    package_dir = os.path.join(DRIVE_PROJECT_DIR, package_name)
    os.makedirs(package_dir, exist_ok=True)

    # Copy corpus files to package directory (without loading into memory)
    convokit_path = os.path.expanduser(f"~/.convokit/saved-corpora/{corpus_output_path}")
    if os.path.exists(convokit_path):
        corpus_dest = os.path.join(package_dir, "corpus_data")
        shutil.copytree(convokit_path, corpus_dest)
        print(f"📋 Corpus files copied to package")

    # Create metadata file
    metadata = {
        'created_timestamp': timestamp,
        'corpus_name': corpus_output_path,
        'conversations_processed': max_conversations or 'all',
        'topic_statistics': topic_stats,
        'usage_instructions': {
            'loading': "Use convokit.Corpus(filename='corpus_data') to load",
            'topic_access': "utterance.meta['thread_topic'] for topic labels",
            'confidence_access': "utterance.meta['topic_confidence'] for confidence scores"
        }
    }

    with open(os.path.join(package_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create README
    readme_content = f"""# Political Discussion Corpus - Topic Annotated

## Dataset Information
- Created: {timestamp}
- Conversations: {max_conversations or 'All available'}
- Source: Reddit r/PoliticalDiscussion via ConvoKit

## Files Included
- corpus_data/: ConvoKit corpus with topic annotations
- metadata.json: Processing statistics and usage information
- README.md: This file

## Usage

```python
import convokit

# Load the corpus
corpus = convokit.Corpus(filename='corpus_data')

# Access topic annotations
for utterance in corpus.iter_utterances():
    topic = utterance.meta['thread_topic']
    confidence = utterance.meta['topic_confidence']
    is_on_topic = utterance.meta['is_on_topic']
    print(f"Topic: {{topic}} (confidence: {{confidence:.3f}})")
```

## Topic Categories
{chr(10).join(f"- {topic}" for topic in political_topics)}

## Statistics
{json.dumps(topic_stats, indent=2)}
"""

    with open(os.path.join(package_dir, "README.md"), 'w') as f:
        f.write(readme_content)

    # Create zip file in Drive directory
    zip_path = os.path.join(DRIVE_PROJECT_DIR, f"{package_name}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files_list in os.walk(package_dir):
            for file in files_list:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, package_dir)
                zipf.write(file_path, arcname)

    print(f"📁 Package saved to Google Drive: {zip_path}")

    # Auto-download to local system (only in Colab)
    if IN_COLAB:
        print("🔽 Starting automatic download...")
        files.download(zip_path)
        print(f"✅ Downloaded: {package_name}.zip")

    # Clean up the unzipped directory to save space
    shutil.rmtree(package_dir)

    return zip_path, package_name


def add_topic_metadata_to_corpus(corpus_path_or_corpus, output_path=None, max_conversations=None,
                                 sample_strategy='first', save_to_drive=True):
    """
    Load corpus, add topic metadata, and save with optional Google Drive backup
    MEMORY EFFICIENT: Original corpus handling preserved
    """
    # Load corpus if path provided, otherwise use existing corpus
    if isinstance(corpus_path_or_corpus, str):
        print(f"Loading corpus from {corpus_path_or_corpus}...")
        # corpus = convokit.Corpus(filename=corpus_path_or_corpus)
        corpus = Corpus(download('subreddit-PoliticalDiscussion'))
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

        # Save subset corpus
        print(f"Saving subset corpus to {output_path}...")
        try:
            subset_corpus.dump(output_path)
            print("✅ Subset corpus saved successfully!")
        except Exception as e:
            print(f"❌ Error saving subset corpus: {e}")
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

    # Calculate topic statistics
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

    # Save topic statistics to ConvoKit directory
    try:
        stats_file = os.path.expanduser(f"~/.convokit/saved-corpora/{output_path}/topic_statistics.json")
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print("✅ Topic statistics saved successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not save statistics: {e}")

    # Create downloadable package and save to Google Drive
    if save_to_drive:
        print("\n🎁 Creating downloadable package...")
        try:
            package_path, package_name = create_downloadable_package(output_path, stats, max_conversations)
            print(f"✅ Package created and {'downloaded' if IN_COLAB else 'saved'}: {package_name}.zip")

            # Also save processing log to Google Drive
            log_content = f"""# Topic Detection Processing Log
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Max conversations: {max_conversations or 'All'}
- Sampling strategy: {sample_strategy}
- Total available: {total_available}
- Processed: {len(conversations_to_process)}

## Results
- Output path: {output_path}
- Final utterances: {len(list(final_corpus.iter_utterances()))}

## Topic Distribution
{json.dumps(topic_counts, indent=2)}

## Files Created
- {'Google Drive' if IN_COLAB else 'Local'}: {package_path}
- ConvoKit corpus: ~/.convokit/saved-corpora/{output_path}
"""

            log_file = os.path.join(DRIVE_PROJECT_DIR, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            with open(log_file, 'w') as f:
                f.write(log_content)
            print(f"📝 Processing log saved: {log_file}")

        except Exception as e:
            print(f"❌ Error creating package: {e}")
            print("Corpus was saved to ConvoKit directory, but package creation failed")

    print("Topic detection completed!")
    print(f"Topic distribution: {topic_counts}")
    print(f"Final output location: ~/.convokit/saved-corpora/{output_path}")
    print(f"{'Google Drive' if IN_COLAB else 'Local'} backup: {DRIVE_PROJECT_DIR}")

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
    preview_results = []

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

        preview_data = {
            'title': display_title,
            'first_post': first_post[:200] + ('...' if len(first_post) > 200 else ''),
            'topic': topic_result['topic'],
            'confidence': topic_result['confidence']
        }
        preview_results.append(preview_data)

        print(f"Title: {display_title}")
        print(f"First Post: {first_post[:200]}{'...' if len(first_post) > 200 else ''}")
        print(f"Predicted Topic: {topic_result['topic']} (confidence: {topic_result['confidence']:.3f})")
        print("-" * 80)

    # Save preview to Google Drive
    preview_file = os.path.join(DRIVE_PROJECT_DIR, f"topic_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(preview_file, 'w') as f:
        json.dump(preview_results, f, indent=2)
    print(f"📄 Preview results saved to: {preview_file}")

    return corpus


# Main execution
if __name__ == "__main__":
    CORPUS_PATH = "/content/drive/MyDrive/temporal-belief-analysis/subreddit-PoliticalDiscussion"

    # Check if corpus exists in Drive, if not provide instructions
    if not os.path.exists(CORPUS_PATH):
        print(f"❌ Corpus not found at: {CORPUS_PATH}")
        print("\n📥 Please upload your ConvoKit corpus to Google Drive:")
        print(f"   1. Upload the corpus folder to: {DRIVE_PROJECT_DIR}")
        print("   2. Make sure it's named 'subreddit-PoliticalDiscussion'")
        print("   3. Restart this script")
        exit()

    print("=== PREVIEW MODE ===")
    loaded_corpus = preview_topic_classification(CORPUS_PATH, num_examples=20)

    print("\n🤖 Does the topic classification look reasonable?")
    response = input("Continue with processing? (y/n): ")

    if response.lower() == 'y':
        print("\n=== PROCESSING OPTIONS ===")
        print("1. 🧪 Test subset (100 conversations) - for testing stance detection")
        print("2. 🔧 Dev subset (500 conversations) - for development")
        print("3. 📊 Large subset (2000 conversations) - for validation")
        print("4. 🌍 Full dataset - process everything")

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
        print("📁 Files will be saved to Google Drive and auto-downloaded")
        print("🔄 Processing started...")

        annotated_corpus = add_topic_metadata_to_corpus(
            loaded_corpus,
            max_conversations=max_conversations,
            sample_strategy=strategy,
            save_to_drive=True
        )

        print("\n🎉 PROCESSING COMPLETE!")
        print("✅ Corpus annotated with topic metadata")
        print("💾 Files saved to Google Drive")
        print("⬇️ Package downloaded to your local system")
        print(f"📂 Google Drive location: {DRIVE_PROJECT_DIR}")

        # Summary
        print(f"\n📋 SUMMARY:")
        if max_conversations:
            print(f"   - Processed: {max_conversations} conversations (subset)")
        else:
            print(f"   - Processed: All conversations (full dataset)")
        print(f"   - Strategy: {strategy}")
        print(f"   - Files ready for stance detection training!")

    else:
        print("🛑 Stopping here. You can adjust the topics list or classification parameters.")


def quick_test_subset():
    """
    Quick function to create a small test subset for development
    """
    CORPUS_PATH = "/content/drive/MyDrive/temporal-belief-analysis/subreddit-PoliticalDiscussion"

    if not os.path.exists(CORPUS_PATH):
        print(f"❌ Corpus not found at: {CORPUS_PATH}")
        return None

    print("🚀 Creating quick test subset (100 conversations)...")
    corpus = convokit.Corpus(filename=CORPUS_PATH)

    return add_topic_metadata_to_corpus(
        corpus,
        max_conversations=100,
        sample_strategy='diverse',
        save_to_drive=True
    )


def verify_downloaded_corpus(zip_path):
    """
    Verify that the downloaded corpus is working correctly
    """
    import zipfile
    import tempfile

    print(f"🔍 Verifying downloaded corpus: {zip_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find corpus directory
        corpus_dir = os.path.join(temp_dir, "corpus_data")
        if not os.path.exists(corpus_dir):
            print("❌ Corpus data not found in zip file")
            return False

        try:
            # Load corpus
            corpus = convokit.Corpus(filename=corpus_dir)

            # Check utterances
            utterances = list(corpus.iter_utterances())[:5]
            print(f"✅ Corpus loaded successfully: {len(utterances)} utterances checked")

            # Check topic metadata
            for utt in utterances:
                topic = utt.meta.get('thread_topic', 'NOT FOUND')
                confidence = utt.meta.get('topic_confidence', 'NOT FOUND')
                print(f"   - Topic: {topic}, Confidence: {confidence}")

            return True

        except Exception as e:
            print(f"❌ Error loading corpus: {e}")
            return False