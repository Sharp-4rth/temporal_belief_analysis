import convokit
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import os
import json
import torch
from datetime import datetime
from collections import defaultdict
import random

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


def find_active_users(corpus, min_comments=10, min_timespan_days=30, max_utterances=10000, max_comments_per_user=300):
    """
    Find users with sufficient activity for temporal analysis
    Memory-efficient version that processes utterances in chunks

    Args:
        corpus: ConvoKit corpus
        min_comments: Minimum number of comments required
        min_timespan_days: Minimum timespan between first and last comment
        max_utterances: Maximum utterances to process (for memory management)
        max_comments_per_user: Maximum comments per user to avoid super-users

    Returns:
        List of user IDs suitable for temporal analysis
    """
    print("Analyzing user activity patterns (memory-efficient mode)...")
    user_activity = defaultdict(list)

    # Convert timestamp function (handles both int and datetime)
    def convert_timestamp(ts):
        if isinstance(ts, int):
            return datetime.fromtimestamp(ts)
        elif hasattr(ts, 'timestamp'):
            return ts
        else:
            return datetime.fromtimestamp(float(ts))

    # Process utterances with memory limit
    processed_count = 0
    print(f"Processing up to {max_utterances:,} utterances for user analysis...")

    for utterance in tqdm(corpus.iter_utterances(), desc="Processing utterances"):
        if processed_count >= max_utterances:
            print(f"Reached processing limit of {max_utterances:,} utterances")
            break

        if utterance.speaker.id != 'AutoModerator':  # Skip bot posts
            try:
                timestamp = convert_timestamp(utterance.timestamp)

                # Skip users who already have too many comments
                if len(user_activity[utterance.speaker.id]) >= max_comments_per_user:
                    continue

                user_activity[utterance.speaker.id].append({
                    'timestamp': timestamp,
                    'conversation_id': utterance.conversation_id,
                    'utterance_id': utterance.id,
                    'text_length': len(utterance.text)
                })
            except Exception as e:
                print(f"Warning: Could not process timestamp for utterance {utterance.id}: {e}")
                continue

        processed_count += 1

    print(f"Processed {processed_count:,} utterances from {len(user_activity)} unique users")

    # Filter users by activity criteria
    suitable_users = []
    for user_id, posts in user_activity.items():
        if min_comments <= len(posts) <= max_comments_per_user:
            # Sort by timestamp to check timespan
            posts_sorted = sorted(posts, key=lambda x: x['timestamp'])
            first_post = posts_sorted[0]['timestamp']
            last_post = posts_sorted[-1]['timestamp']

            timespan_days = (last_post - first_post).days

            if timespan_days >= min_timespan_days:
                suitable_users.append({
                    'user_id': user_id,
                    'total_comments': len(posts),
                    'timespan_days': timespan_days,
                    'avg_text_length': sum(p['text_length'] for p in posts) / len(posts),
                    'first_post': first_post,
                    'last_post': last_post
                })

    # Sort by activity level (comments * timespan)
    suitable_users.sort(key=lambda x: x['total_comments'] * x['timespan_days'], reverse=True)

    print(f"Found {len(suitable_users)} users meeting criteria:")
    print(f"  - Min {min_comments} comments, Max {max_comments_per_user} comments")
    print(f"  - Min {min_timespan_days} days active")
    print(f"  - Processed {processed_count:,} total utterances")

    return suitable_users


def sample_user_threads(corpus, user_id, max_threads=50, sampling_strategy='temporal_diverse'):
    """
    Sample threads that a specific user participated in
    Memory-efficient version with timestamp handling

    Args:
        corpus: ConvoKit corpus
        user_id: Target user ID
        max_threads: Maximum number of threads to sample
        sampling_strategy: 'temporal_diverse', 'recent', 'random', 'chronological'

    Returns:
        List of conversation IDs and metadata
    """

    # Convert timestamp function
    def convert_timestamp(ts):
        if isinstance(ts, int):
            return datetime.fromtimestamp(ts)
        elif hasattr(ts, 'timestamp'):
            return ts
        else:
            return datetime.fromtimestamp(float(ts))

    # Get all conversations this user participated in
    user_conversations = []

    for utterance in corpus.iter_utterances():
        if utterance.speaker.id == user_id:
            try:
                timestamp = convert_timestamp(utterance.timestamp)
                conv = corpus.get_conversation(utterance.conversation_id)
                user_conversations.append({
                    'conversation_id': utterance.conversation_id,
                    'user_timestamp': timestamp,
                    'conversation': conv,
                    'user_utterance_id': utterance.id,
                    'user_text': utterance.text
                })
            except Exception as e:
                print(f"Warning: Could not process utterance {utterance.id} for user {user_id}: {e}")
                continue

    # Remove duplicates (user might have multiple comments in same thread)
    unique_conversations = {}
    for conv_data in user_conversations:
        conv_id = conv_data['conversation_id']
        if conv_id not in unique_conversations:
            unique_conversations[conv_id] = conv_data
        else:
            # Keep the earliest timestamp for this user in this conversation
            if conv_data['user_timestamp'] < unique_conversations[conv_id]['user_timestamp']:
                unique_conversations[conv_id] = conv_data

    user_conversations = list(unique_conversations.values())

    print(f"User {user_id} participated in {len(user_conversations)} unique conversations")

    # Apply sampling strategy
    if len(user_conversations) <= max_threads:
        sampled_conversations = user_conversations
    else:
        if sampling_strategy == 'temporal_diverse':
            # Sort by timestamp and take evenly spaced samples
            sorted_convs = sorted(user_conversations, key=lambda x: x['user_timestamp'])
            step = len(sorted_convs) // max_threads
            sampled_conversations = sorted_convs[::step][:max_threads]

        elif sampling_strategy == 'recent':
            # Take most recent threads
            sorted_convs = sorted(user_conversations, key=lambda x: x['user_timestamp'], reverse=True)
            sampled_conversations = sorted_convs[:max_threads]

        elif sampling_strategy == 'chronological':
            # Take earliest threads chronologically
            sorted_convs = sorted(user_conversations, key=lambda x: x['user_timestamp'])
            sampled_conversations = sorted_convs[:max_threads]

        elif sampling_strategy == 'random':
            sampled_conversations = random.sample(user_conversations, max_threads)

        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    print(f"Sampled {len(sampled_conversations)} conversations using '{sampling_strategy}' strategy")

    return sampled_conversations


def create_user_centric_corpus(corpus_or_loaded_corpus,
                               num_users=10,
                               threads_per_user=50,
                               user_selection='most_active',
                               thread_sampling='temporal_diverse',
                               output_path=None,
                               suitable_users=None):
    """
    Create a subset corpus following specific users through their comment history

    Args:
        corpus_or_loaded_corpus: Path to corpus OR already-loaded corpus
        num_users: Number of users to track
        threads_per_user: Number of threads to sample per user
        user_selection: 'most_active', 'diverse_activity', 'random'
        thread_sampling: 'temporal_diverse', 'recent', 'chronological', 'random'
        output_path: Where to save the subset corpus
        suitable_users: Pre-found suitable users (to avoid re-analysis)

    Returns:
        Processed corpus with topic annotations
    """
    # Load corpus if path provided, otherwise use already loaded corpus
    if isinstance(corpus_or_loaded_corpus, str):
        print(f"Loading corpus from {corpus_or_loaded_corpus}...")
        corpus = convokit.Corpus(filename=corpus_or_loaded_corpus)
    else:
        print("Using already-loaded corpus...")
        corpus = corpus_or_loaded_corpus

    # Create output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"PoliticalDiscussion_users_{num_users}_threads_{threads_per_user}_{timestamp}"

    print(f"Creating user-centric corpus: {num_users} users × {threads_per_user} threads each")
    print(f"Will save to: {output_path}")

    # Step 1: Use provided suitable users or find them
    if suitable_users is None:
        print("Finding suitable users for temporal analysis...")
        max_utterances_to_process = min(10000, len(list(corpus.iter_utterances())))  # Process up to 10K utterances
        suitable_users = find_active_users(corpus, min_comments=threads_per_user, min_timespan_days=30,
                                           max_utterances=max_utterances_to_process, max_comments_per_user=300)
    else:
        print("Using pre-analyzed suitable users...")

    if len(suitable_users) < num_users:
        print(f"⚠️  Only found {len(suitable_users)} suitable users, less than requested {num_users}")
        num_users = len(suitable_users)

    # Step 2: Select users based on strategy
    if user_selection == 'most_active':
        selected_users = suitable_users[:num_users]
    elif user_selection == 'diverse_activity':
        # Select users with different activity patterns
        step = len(suitable_users) // num_users
        selected_users = suitable_users[::step][:num_users]
    elif user_selection == 'random':
        selected_users = random.sample(suitable_users, num_users)
    else:
        raise ValueError(f"Unknown user selection strategy: {user_selection}")

    print(f"\nSelected {len(selected_users)} users:")
    for i, user in enumerate(selected_users[:5]):  # Show first 5
        print(f"  {i + 1}. User {user['user_id']}: {user['total_comments']} comments over {user['timespan_days']} days")
    if len(selected_users) > 5:
        print(f"  ... and {len(selected_users) - 5} more")

    # Step 3: Sample threads for each user and classify topics
    all_conversations_to_include = set()
    user_conversation_map = {}  # Track which conversations belong to which users
    conversation_topics = {}

    for user_data in tqdm(selected_users, desc="Processing users"):
        user_id = user_data['user_id']

        # Sample conversations for this user
        user_conversations = sample_user_threads(
            corpus, user_id,
            max_threads=threads_per_user,
            sampling_strategy=thread_sampling
        )

        # Track conversations for this user
        user_conversation_map[user_id] = []

        # Classify topics for user's conversations
        print(f"\nClassifying topics for user {user_id}...")
        for conv_data in tqdm(user_conversations, desc=f"User {user_id} threads"):
            conversation_id = conv_data['conversation_id']
            conversation = conv_data['conversation']

            # Get thread title and first post for classification
            title = conversation.meta.get('title', '')
            first_utterance = conversation.get_chronological_utterance_list()[0]
            first_post = first_utterance.text

            # Classify topic
            combined_text = f"{title}. {first_post}"[:500]
            if not title and len(first_post.strip()) < 20:
                combined_text = "general discussion"

            topic_result = classify_thread_topic(combined_text)

            # Store results
            conversation_topics[conversation_id] = topic_result
            all_conversations_to_include.add(conversation_id)

            user_conversation_map[user_id].append({
                'conversation_id': conversation_id,
                'user_timestamp': conv_data['user_timestamp'],
                'topic': topic_result['topic'],
                'topic_confidence': topic_result['confidence'],
                'user_utterance_id': conv_data['user_utterance_id']
            })

    print(f"\nTotal unique conversations to include: {len(all_conversations_to_include)}")

    # Step 4: Create subset corpus with selected conversations
    print("Creating subset corpus...")
    all_utterances = list(corpus.iter_utterances())
    utterances_to_keep = [utt for utt in all_utterances
                          if utt.conversation_id in all_conversations_to_include]

    print(f"Keeping {len(utterances_to_keep)} utterances from {len(all_conversations_to_include)} conversations")

    # Create subset corpus
    subset_corpus = convokit.Corpus(utterances=utterances_to_keep)

    # Step 5: Add metadata to all utterances
    print("Adding metadata to utterances...")
    for utterance in tqdm(subset_corpus.iter_utterances(), desc="Adding metadata"):
        conv_id = utterance.conversation_id

        # Add topic metadata
        if conv_id in conversation_topics:
            topic_data = conversation_topics[conv_id]
            utterance.meta['thread_topic'] = topic_data['topic']
            utterance.meta['topic_confidence'] = topic_data['confidence']
            utterance.meta['is_on_topic'] = topic_data['confidence'] > 0.3
        else:
            utterance.meta['thread_topic'] = 'unknown'
            utterance.meta['topic_confidence'] = 0.0
            utterance.meta['is_on_topic'] = False

        # Add user tracking metadata
        utterance.meta['is_tracked_user'] = utterance.speaker.id in user_conversation_map

        if utterance.speaker.id in user_conversation_map:
            # Find this user's entry for this conversation
            user_convs = user_conversation_map[utterance.speaker.id]
            user_conv_data = next((uc for uc in user_convs if uc['conversation_id'] == conv_id), None)

            if user_conv_data:
                utterance.meta['user_sequence_timestamp'] = user_conv_data['user_timestamp']
                utterance.meta['is_user_primary_utterance'] = (utterance.id == user_conv_data['user_utterance_id'])
            else:
                utterance.meta['user_sequence_timestamp'] = None
                utterance.meta['is_user_primary_utterance'] = False
        else:
            utterance.meta['user_sequence_timestamp'] = None
            utterance.meta['is_user_primary_utterance'] = False

    # Step 6: Save corpus
    print(f"Saving user-centric corpus to {output_path}...")
    try:
        subset_corpus.dump(output_path)
        print("✅ User-centric corpus saved successfully!")
    except Exception as e:
        print(f"❌ Error saving corpus: {e}")
        alt_path = f"corpus_backup_{datetime.now().strftime('%H%M%S')}"
        print(f"Trying alternative path: {alt_path}")
        subset_corpus.dump(alt_path)
        output_path = alt_path
        print(f"✅ Saved to alternative path: {alt_path}")

    # Step 7: Save detailed statistics
    try:
        stats_file = os.path.expanduser(f"~/.convokit/saved-corpora/{output_path}/user_centric_statistics.json")

        # Calculate topic distribution
        topic_counts = {}
        for topic_data in conversation_topics.values():
            topic = topic_data['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Calculate user statistics
        user_stats = {}
        for user_id, conversations in user_conversation_map.items():
            user_topics = [conv['topic'] for conv in conversations]
            user_topic_counts = {}
            for topic in user_topics:
                user_topic_counts[topic] = user_topic_counts.get(topic, 0) + 1

            user_stats[user_id] = {
                'total_conversations': len(conversations),
                'topic_distribution': user_topic_counts,
                'timespan': {
                    'first': min(conv['user_timestamp'] for conv in conversations).isoformat(),
                    'last': max(conv['user_timestamp'] for conv in conversations).isoformat(),
                    'days': (max(conv['user_timestamp'] for conv in conversations) -
                             min(conv['user_timestamp'] for conv in conversations)).days
                }
            }

        stats = {
            'creation_timestamp': datetime.now().isoformat(),
            'corpus_parameters': {
                'num_users_requested': num_users,
                'threads_per_user': threads_per_user,
                'user_selection_strategy': user_selection,
                'thread_sampling_strategy': thread_sampling
            },
            'corpus_statistics': {
                'total_users_tracked': len(selected_users),
                'total_conversations': len(all_conversations_to_include),
                'total_utterances': len(utterances_to_keep),
                'topic_distribution': topic_counts
            },
            'user_statistics': user_stats,
            'tracked_users': [user['user_id'] for user in selected_users]
        }

        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        print("✅ Detailed statistics saved!")
        print(f"Stats saved to: {stats_file}")

    except Exception as e:
        print(f"⚠️ Warning: Could not save statistics: {e}")

    print("\n🎯 User-centric corpus creation completed!")
    print(f"📊 Final corpus stats:")
    print(f"   - {len(selected_users)} users tracked")
    print(f"   - {len(all_conversations_to_include)} conversations")
    print(f"   - {len(utterances_to_keep)} utterances")
    print(f"   - Topic distribution: {topic_counts}")

    return subset_corpus


def preview_user_activity(corpus_path, num_users=5, max_utterances=10000):
    """
    Preview user activity to help decide sampling parameters
    Memory-efficient version
    """
    print("Loading corpus for user activity preview...")
    corpus = convokit.Corpus(filename=corpus_path)

    suitable_users = find_active_users(corpus, min_comments=5, min_timespan_days=7, max_utterances=max_utterances,
                                       max_comments_per_user=300)

    print(f"\n📊 Top {num_users} Most Active Users:")
    print("-" * 100)

    for i, user in enumerate(suitable_users[:num_users]):
        print(f"{i + 1}. User: {user['user_id']}")
        print(f"   📝 Comments: {user['total_comments']}")
        print(f"   📅 Active span: {user['timespan_days']} days")
        print(f"   📏 Avg comment length: {user['avg_text_length']:.0f} chars")
        print(f"   🕐 Period: {user['first_post'].strftime('%Y-%m-%d')} to {user['last_post'].strftime('%Y-%m-%d')}")

        # Show sample conversation topics for this user
        try:
            sample_conversations = sample_user_threads(corpus, user['user_id'], max_threads=5,
                                                       sampling_strategy='recent')
            print(f"   🗣️  Recent conversation topics:")
            for j, conv_data in enumerate(sample_conversations):
                conv = conv_data['conversation']
                title = conv.meta.get('title', 'No title')
                if title == 'No title':
                    first_post = conv.get_chronological_utterance_list()[0].text
                    title = first_post[:60] + "..." if len(first_post) > 60 else first_post
                print(f"      {j + 1}. {title}")
        except Exception as e:
            print(f"   🗣️  Could not load recent conversations: {e}")
        print("-" * 100)

    return corpus


# Main execution functions
def create_small_dev_corpus(corpus_path, users=5, threads=20):
    """Quick function to create a small development corpus"""
    return create_user_centric_corpus(
        corpus_path,
        num_users=users,
        threads_per_user=threads,
        user_selection='most_active',
        thread_sampling='temporal_diverse'
    )


def create_medium_analysis_corpus(corpus_path, users=20, threads=50):
    """Create a medium-sized corpus for initial analysis"""
    return create_user_centric_corpus(
        corpus_path,
        num_users=users,
        threads_per_user=threads,
        user_selection='diverse_activity',
        thread_sampling='temporal_diverse'
    )


def create_full_research_corpus(corpus_path, users=100, threads=50):
    """Create a full research corpus for complete analysis"""
    return create_user_centric_corpus(
        corpus_path,
        num_users=users,
        threads_per_user=threads,
        user_selection='diverse_activity',
        thread_sampling='temporal_diverse'
    )


# Verification function
def analyze_user_sequences(corpus_path):
    """
    Analyze the temporal sequences created for each user
    """
    corpus = convokit.Corpus(filename=corpus_path)

    print("Analyzing user temporal sequences...")

    # Group utterances by tracked users
    user_sequences = defaultdict(list)

    for utterance in corpus.iter_utterances():
        if utterance.meta.get('is_tracked_user', False):
            user_sequences[utterance.speaker.id].append({
                'timestamp': utterance.meta.get('user_sequence_timestamp', utterance.timestamp),
                'conversation_id': utterance.conversation_id,
                'topic': utterance.meta.get('thread_topic', 'unknown'),
                'topic_confidence': utterance.meta.get('topic_confidence', 0.0),
                'is_primary': utterance.meta.get('is_user_primary_utterance', False),
                'text': utterance.text
            })

    print(f"\n📊 Temporal Sequence Analysis:")
    print(f"Found sequences for {len(user_sequences)} users")

    for user_id, sequence in list(user_sequences.items())[:3]:  # Show first 3 users
        # Sort by timestamp
        sequence.sort(key=lambda x: x['timestamp'])

        print(f"\n👤 User: {user_id}")
        print(f"   📝 Total comments in corpus: {len(sequence)}")
        print(
            f"   📅 Timespan: {sequence[0]['timestamp'].strftime('%Y-%m-%d')} to {sequence[-1]['timestamp'].strftime('%Y-%m-%d')}")

        # Show topic distribution
        topics = [s['topic'] for s in sequence]
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        print(f"   🏷️  Topics discussed: {dict(list(topic_counts.items())[:5])}")

        # Show temporal sequence (first 5 comments)
        print(f"   ⏰ First 5 comments in sequence:")
        for i, comment in enumerate(sequence[:5]):
            print(
                f"      {i + 1}. {comment['timestamp'].strftime('%m-%d')} | {comment['topic'][:15]} | {comment['text'][:50]}...")

    return user_sequences


if __name__ == "__main__":
    CORPUS_PATH = "../../../data/subreddit-PoliticalDiscussion"

    print("=== USER-CENTRIC CORPUS CREATION ===")
    print("This creates a corpus following specific users through their comment history")
    print()

    # Get parameters FIRST before loading anything
    print("=== ANALYSIS PARAMETERS ===")
    print("Recommended settings for testing: 5 users, 20 threads each, 300 max comments per user")
    print()

    try:
        num_users = int(input("Number of users to track (default 5): ") or "5")
        threads_per_user = int(input("Threads per user (default 20): ") or "20")
        max_comments_per_user = int(input("Max comments per user (default 300): ") or "300")
        max_utterances = int(input("Max utterances to process for user analysis (default 10000): ") or "10000")

        print(f"\n📋 Your settings:")
        print(f"   - {num_users} users")
        print(f"   - {threads_per_user} threads per user")
        print(f"   - Max {max_comments_per_user} comments per user")
        print(f"   - Process {max_utterances:,} utterances for analysis")

    except ValueError:
        print("Invalid input, using defaults: 5 users, 20 threads, 300 max comments, 10K utterances")
        num_users = 5
        threads_per_user = 20
        max_comments_per_user = 300
        max_utterances = 10000

    proceed = input(f"\nProceed with these settings? (y/n): ")
    if proceed.lower() != 'y':
        print("Exiting. Run again with different parameters.")
        exit()

    # NOW load corpus once and do everything
    print(f"\n⏳ Loading corpus and analyzing user activity...")
    print(f"Processing {max_utterances:,} utterances to find suitable users...")

    loaded_corpus = convokit.Corpus(filename=CORPUS_PATH)
    # corpus = Corpus(download('subreddit-PoliticalDiscussion'))

    # Find users with the specified parameters
    suitable_users = find_active_users(
        loaded_corpus,
        min_comments=threads_per_user,
        min_timespan_days=30,
        max_utterances=max_utterances,
        max_comments_per_user=max_comments_per_user
    )

    if len(suitable_users) < num_users:
        print(f"⚠️  Only found {len(suitable_users)} suitable users, less than requested {num_users}")
        print("Options:")
        print("1. Continue with fewer users")
        print("2. Reduce min threads per user")
        print("3. Increase max utterances to process")

        choice = input("Choose (1/2/3): ")
        if choice == '2':
            new_threads = int(input("New min threads per user: "))
            suitable_users = find_active_users(
                loaded_corpus,
                min_comments=new_threads,
                min_timespan_days=30,
                max_utterances=max_utterances,
                max_comments_per_user=max_comments_per_user
            )
        elif choice == '3':
            new_max = int(input("New max utterances to process: "))
            suitable_users = find_active_users(
                loaded_corpus,
                min_comments=threads_per_user,
                min_timespan_days=30,
                max_utterances=new_max,
                max_comments_per_user=max_comments_per_user
            )
        # If choice == '1', just continue with fewer users

    # Show the users we found
    print(f"\n📊 Found {len(suitable_users)} suitable users:")
    print("-" * 80)

    users_to_show = min(num_users, len(suitable_users))
    for i, user in enumerate(suitable_users[:users_to_show]):
        print(f"{i + 1}. User: {user['user_id']}")
        print(f"   📝 Comments: {user['total_comments']}")
        print(f"   📅 Active span: {user['timespan_days']} days")
        print(f"   📏 Avg comment length: {user['avg_text_length']:.0f} chars")
        print(f"   🕐 Period: {user['first_post'].strftime('%Y-%m-%d')} to {user['last_post'].strftime('%Y-%m-%d')}")
        print("-" * 80)

    final_proceed = input(f"\nProceed with creating corpus using these {users_to_show} users? (y/n): ")

    if final_proceed.lower() == 'y':
        print(f"\n🚀 Creating user-centric corpus...")

        # Use the already-loaded corpus and already-found users
        corpus = create_user_centric_corpus(
            loaded_corpus,  # Pass the loaded corpus, not the path
            num_users=users_to_show,
            threads_per_user=threads_per_user,
            user_selection='most_active',  # Since we already have the sorted list
            thread_sampling='temporal_diverse'
        )

        print("\n✅ Corpus creation completed!")
        print("Files saved to ~/.convokit/saved-corpora/")
        print("\nNext step: Apply stance detection to this user-centric corpus")

    else:
        print("Stopping here. Run again with different parameters.")


# Remove the old main execution block
def old_main_for_reference():
    """
    Old main execution - kept for reference but not used
    """
    pass