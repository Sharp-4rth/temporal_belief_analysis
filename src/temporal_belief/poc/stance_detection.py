import convokit
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import os
import json
import torch
import numpy as np
from datetime import datetime

# Initialize multiple models for ensemble (optional - comment out if too slow)
print("Loading models for stance classification...")
device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else -1

# Primary model: BART (your current choice)
bart_classifier = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli",
                           device=device)

# Alternative model: DeBERTa-v3 (research shows better performance)
# Uncomment this if you want to try the ensemble approach
# deberta_classifier = pipeline("zero-shot-classification",
#                               model="microsoft/deberta-v3-large-mnli",
#                               device=device)

# TOPIC-SPECIFIC STANCE HYPOTHESES (SPINOS-style)
# Each topic gets its own specific stance categories with intensity levels

TOPIC_STANCE_HYPOTHESES = {
    'abortion and reproductive rights': {
        'strongly_favor': [
            'The author strongly supports abortion rights and reproductive freedom',
            'This comment advocates for unrestricted access to abortion services',
            'The author expresses strong pro-choice activist viewpoints'
        ],
        'moderately_favor': [
            'The author moderately supports abortion rights with some limitations',
            'This comment leans toward pro-choice but accepts some restrictions',
            'The author supports reproductive rights with reasonable regulations'
        ],
        'neutral': [
            'This comment provides neutral information about abortion policy',
            'The author presents balanced views on reproductive rights',
            'This text discusses abortion without taking a clear stance'
        ],
        'moderately_against': [
            'The author has concerns about abortion but accepts some circumstances',
            'This comment leans pro-life but allows for exceptions',
            'The author moderately opposes abortion with some flexibility'
        ],
        'strongly_against': [
            'The author strongly opposes abortion and supports pro-life positions',
            'This comment advocates for complete protection of unborn life',
            'The author expresses strong anti-abortion activist viewpoints'
        ]
    },

    'gun rights and control': {
        'strongly_favor': [
            'The author strongly supports gun rights and Second Amendment freedoms',
            'This comment advocates for unrestricted firearm ownership rights',
            'The author expresses strong pro-gun activist viewpoints'
        ],
        'moderately_favor': [
            'The author supports gun rights but accepts some safety regulations',
            'This comment leans pro-gun but allows reasonable restrictions',
            'The author favors gun ownership with common-sense limitations'
        ],
        'neutral': [
            'This comment provides neutral information about gun policy',
            'The author presents balanced views on firearm regulations',
            'This text discusses guns without taking a clear stance'
        ],
        'moderately_against': [
            'The author supports gun control but respects some ownership rights',
            'This comment leans toward restrictions but allows some gun rights',
            'The author moderately favors gun control with exceptions'
        ],
        'strongly_against': [
            'The author strongly supports gun control and firearm restrictions',
            'This comment advocates for strict limitations on gun ownership',
            'The author expresses strong gun control activist viewpoints'
        ]
    },

    'immigration': {
        'strongly_favor': [
            'The author strongly supports immigration and open border policies',
            'This comment advocates for expanded immigration and refugee rights',
            'The author expresses strong pro-immigration activist viewpoints'
        ],
        'moderately_favor': [
            'The author supports immigration with reasonable processing systems',
            'This comment leans pro-immigration but accepts some controls',
            'The author favors welcoming immigrants with proper procedures'
        ],
        'neutral': [
            'This comment provides neutral information about immigration policy',
            'The author presents balanced views on immigration reform',
            'This text discusses immigration without taking a clear stance'
        ],
        'moderately_against': [
            'The author has concerns about immigration but supports legal pathways',
            'This comment leans toward restrictions but allows controlled immigration',
            'The author moderately opposes unchecked immigration'
        ],
        'strongly_against': [
            'The author strongly opposes immigration and supports border restrictions',
            'This comment advocates for strict immigration controls and enforcement',
            'The author expresses strong anti-immigration viewpoints'
        ]
    },

    'healthcare': {
        'strongly_favor': [
            'The author strongly supports universal healthcare and single-payer systems',
            'This comment advocates for government-provided healthcare for all citizens',
            'The author expresses strong support for socialized medicine and healthcare as a right'
        ],
        'moderately_favor': [
            'The author supports expanded government healthcare with some private options',
            'This comment leans toward universal coverage but accepts mixed public-private systems',
            'The author favors government healthcare expansion with pragmatic compromises'
        ],
        'neutral': [
            'This comment provides neutral information about healthcare policy without taking sides',
            'The author presents balanced views on different healthcare systems',
            'This text discusses healthcare options without advocating for specific approaches'
        ],
        'moderately_against': [
            'The author prefers market-based healthcare but accepts some government safety nets',
            'This comment leans toward private healthcare while acknowledging need for limited government role',
            'The author moderately opposes universal healthcare but supports targeted government programs'
        ],
        'strongly_against': [
            'The author strongly opposes government healthcare and advocates for free-market solutions',
            'This comment advocates against socialized medicine and for private healthcare systems',
            'The author expresses strong opposition to universal healthcare and government involvement'
        ]
    },

    'climate change': {
        'strongly_favor': [
            'The author strongly supports aggressive climate action and environmental protection',
            'This comment advocates for immediate action on climate change',
            'The author expresses strong environmental activist viewpoints'
        ],
        'moderately_favor': [
            'The author supports climate action with balanced economic considerations',
            'This comment leans toward environmental protection with practical limits',
            'The author favors climate policies with gradual implementation'
        ],
        'neutral': [
            'This comment provides neutral information about climate policy',
            'The author presents balanced views on environmental issues',
            'This text discusses climate without taking a clear stance'
        ],
        'moderately_against': [
            'The author questions some climate policies but accepts environmental concerns',
            'This comment leans skeptical but allows for some climate action',
            'The author moderately opposes aggressive climate regulations'
        ],
        'strongly_against': [
            'The author strongly opposes climate regulations and questions climate science',
            'This comment advocates against environmental restrictions on business',
            'The author expresses strong climate skepticism and anti-regulation views'
        ]
    }
}

# General fallback hypotheses for unknown topics
GENERAL_STANCE_HYPOTHESES = {
    'strongly_favor': [
        'The author strongly supports the main position being discussed',
        'This comment advocates strongly for the topic being debated',
        'The author expresses strong support for the primary viewpoint'
    ],
    'moderately_favor': [
        'The author moderately supports the main position with some reservations',
        'This comment leans toward support but with qualifications',
        'The author generally favors the position but acknowledges concerns'
    ],
    'neutral': [
        'This comment provides neutral information without taking sides',
        'The author presents balanced views on the topic',
        'This text discusses the issue without clear position advocacy'
    ],
    'moderately_against': [
        'The author has concerns about the main position but shows some flexibility',
        'This comment leans against the primary viewpoint with some exceptions',
        'The author moderately opposes the position but acknowledges some merit'
    ],
    'strongly_against': [
        'The author strongly opposes the main position being discussed',
        'This comment advocates strongly against the topic being debated',
        'The author expresses strong opposition to the primary viewpoint'
    ]
}

# Simplified stance categories for quick classification
SIMPLE_STANCE_CATEGORIES = [
    'strongly_favor',
    'moderately_favor',
    'neutral',
    'moderately_against',
    'strongly_against'
]


def get_stance_hypotheses_for_topic(topic):
    """
    Get topic-specific stance hypotheses in SPINOS style

    Args:
        topic (str): The discussion topic

    Returns:
        dict: Topic-appropriate stance categories and hypotheses
    """
    # Normalize topic name for lookup
    topic_lower = topic.lower() if topic else ""

    # Topic mapping with flexible matching
    if any(keyword in topic_lower for keyword in ['abortion', 'reproductive', 'pro-choice', 'pro-life']):
        return TOPIC_STANCE_HYPOTHESES['abortion and reproductive rights']
    elif any(keyword in topic_lower for keyword in ['gun', 'firearm', 'second amendment', 'weapon']):
        return TOPIC_STANCE_HYPOTHESES['gun rights and control']
    elif any(keyword in topic_lower for keyword in ['immigration', 'border', 'refugee', 'immigrant']):
        return TOPIC_STANCE_HYPOTHESES['immigration']
    elif any(keyword in topic_lower for keyword in ['healthcare', 'medical', 'insurance', 'medicare']):
        return TOPIC_STANCE_HYPOTHESES['healthcare']
    elif any(keyword in topic_lower for keyword in ['climate', 'environment', 'carbon', 'global warming']):
        return TOPIC_STANCE_HYPOTHESES['climate change']
    else:
        # Use general hypotheses for unknown topics
        return GENERAL_STANCE_HYPOTHESES


def classify_political_stance_improved(text, confidence_threshold=0.25, method='multi_template', topic=None):
    """
    IMPROVED stance classification using SPINOS-style topic-specific categories:
    1. Topic-specific stance categories (pro-abortion vs anti-abortion, not liberal vs conservative)
    2. Multiple hypothesis templates per category
    3. Context-aware prompting
    4. Sensitivity analysis across templates

    Args:
        text (str): The comment text to classify
        confidence_threshold (float): Minimum confidence to assign a stance
        method (str): 'multi_template', 'simple', or 'ensemble'
        topic (str): The discussion topic for context

    Returns:
        dict: Enhanced stance classification results with topic-specific stances
    """
    if not text or len(text.strip()) < 10:
        return {
            'stance': 'neutral',
            'confidence': 0.0,
            'all_scores': {},
            'method_used': method,
            'template_consistency': 0.0,
            'topic_context': topic
        }

    # Clean and prepare text with context
    clean_text = text.strip()
    if len(clean_text) > 500:  # Limit for better performance
        clean_text = clean_text[:500] + "..."

    # Get topic-appropriate stance categories
    stance_hypotheses = get_stance_hypotheses_for_topic(topic)

    # Add contextual framing based on topic
    if topic:
        contextual_text = f"In a discussion about {topic}, this comment states: {clean_text}"
    else:
        contextual_text = f"In a political discussion, this comment states: {clean_text}"

    try:
        if method == 'multi_template':
            return _classify_multi_template_spinos(contextual_text, stance_hypotheses, confidence_threshold, topic)
        elif method == 'simple':
            return _classify_simple_spinos(contextual_text, stance_hypotheses, confidence_threshold, topic)
        elif method == 'ensemble':
            return _classify_ensemble_spinos(contextual_text, stance_hypotheses, confidence_threshold, topic)
        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        print(f"Error classifying stance for text: {str(e)}")
        return {
            'stance': 'error',
            'confidence': 0.0,
            'all_scores': {},
            'method_used': method,
            'template_consistency': 0.0,
            'topic_context': topic
        }


def _classify_multi_template_spinos(text, stance_hypotheses, confidence_threshold, topic):
    """Use multiple hypothesis templates with SPINOS-style topic-specific stances"""
    stance_results = {}
    template_consistency_scores = {}

    for stance, hypotheses in stance_hypotheses.items():
        stance_scores = []

        # Test each hypothesis template for this stance
        for hypothesis in hypotheses:
            result = bart_classifier(text, [hypothesis])
            # Get score for this specific hypothesis
            stance_scores.append(result['scores'][0] if result['labels'][0] == hypothesis else 0)

        # Average across templates for this stance
        avg_confidence = np.mean(stance_scores)
        stance_results[stance] = avg_confidence

        # Measure consistency across templates (lower std = more consistent)
        template_consistency_scores[stance] = 1.0 - (np.std(stance_scores) / (np.mean(stance_scores) + 1e-8))

    # Find best stance
    best_stance = max(stance_results.keys(), key=lambda k: stance_results[k])
    best_confidence = stance_results[best_stance]
    overall_consistency = np.mean(list(template_consistency_scores.values()))

    # Apply confidence threshold
    if best_confidence < confidence_threshold:
        best_stance = 'neutral'
        best_confidence = stance_results.get('neutral', 0.0)

    return {
        'stance': best_stance,
        'confidence': best_confidence,
        'all_scores': stance_results,
        'method_used': 'multi_template_spinos',
        'template_consistency': overall_consistency,
        'reliable': best_confidence > confidence_threshold and overall_consistency > 0.7,
        'topic_context': topic
    }


def _classify_simple_spinos(text, stance_hypotheses, confidence_threshold, topic):
    """Use simple hypothesis templates with SPINOS-style categories"""
    stance_labels = list(stance_hypotheses.keys())
    result = bart_classifier(text, stance_labels)
    top_confidence = result['scores'][0]

    if top_confidence < confidence_threshold:
        stance = 'neutral'
    else:
        stance = result['labels'][0]

    return {
        'stance': stance,
        'confidence': top_confidence,
        'all_scores': dict(zip(result['labels'], result['scores'])),
        'method_used': 'simple_spinos',
        'template_consistency': 1.0,  # N/A for simple method
        'reliable': top_confidence > confidence_threshold,
        'topic_context': topic
    }


def _classify_ensemble_spinos(text, stance_hypotheses, confidence_threshold, topic):
    """Use ensemble approach with SPINOS-style categories"""
    try:
        stance_labels = list(stance_hypotheses.keys())

        # Get results from BART model
        bart_result = bart_classifier(text, stance_labels)
        # If you have DeBERTa uncommented, add it here:
        # deberta_result = deberta_classifier(text, stance_labels)

        # For now, just use BART (expand when ensemble is available)
        bart_probs = np.array(bart_result['scores'])
        ensemble_probs = bart_probs  # Fallback to BART only

        best_idx = np.argmax(ensemble_probs)
        best_stance = stance_labels[best_idx]
        best_confidence = ensemble_probs[best_idx]

        if best_confidence < confidence_threshold:
            best_stance = 'neutral'

        return {
            'stance': best_stance,
            'confidence': best_confidence,
            'all_scores': dict(zip(stance_labels, ensemble_probs)),
            'method_used': 'ensemble_spinos',
            'template_consistency': 1.0,  # N/A for ensemble
            'reliable': best_confidence > confidence_threshold,
            'topic_context': topic
        }

    except Exception as e:
        print(f"Ensemble failed, falling back to simple SPINOS: {e}")
        return _classify_simple_spinos(text, stance_hypotheses, confidence_threshold, topic)


def classify_with_topic_context(text, topic=None, confidence_threshold=0.25):
    """
    SPINOS-style classification that uses topic-specific stance categories
    This is now the main topic-aware classification function
    """
    if not text or len(text.strip()) < 10:
        return classify_political_stance_improved(text, confidence_threshold, topic=topic)

    # Get topic-specific stance hypotheses
    stance_hypotheses = get_stance_hypotheses_for_topic(topic)

    # Topic-specific contextual framing
    if topic and topic != 'general_discussion':
        contextual_text = f"In a discussion about {topic}, this comment states: {text.strip()}"
    else:
        contextual_text = f"In a political discussion, this comment states: {text.strip()}"

    # Use the enhanced classification with topic-specific stances
    result = _classify_multi_template_spinos(contextual_text, stance_hypotheses, confidence_threshold, topic)
    return result


def preview_stance_classification_comparison(corpus_path, num_examples=20, topic_filter=None,
                                             min_text_length=100, max_text_display=800):
    """
    FIXED: Compare different stance classification methods side by side with SPINOS-style topic stances

    Args:
        corpus_path: Path to the ConvoKit corpus
        num_examples: Number of examples to show
        topic_filter: Filter by specific topic (optional)
        min_text_length: Minimum character length for text to be included
        max_text_display: Maximum characters to display for readability
    """
    print("Loading corpus for SPINOS-style stance classification method comparison...")
    corpus = convokit.Corpus(filename=corpus_path)

    print(f"\nComparing SPINOS-style stance classification methods on {num_examples} comments:")
    print(f"Using topic-specific stances (pro-abortion, anti-gun, etc.) instead of political labels")
    print(f"Minimum text length: {min_text_length} characters")
    print(f"Maximum text display: {max_text_display} characters")
    if topic_filter:
        print(f"Filtering for topic: {topic_filter}")
    print("=" * 140)

    utterances = list(corpus.iter_utterances())

    # Filter by topic if specified
    if topic_filter:
        utterances = [u for u in utterances if u.meta.get('thread_topic') == topic_filter]
        print(f"Found {len(utterances)} utterances with topic '{topic_filter}'")

    # Filter by minimum text length to get substantial comments
    substantial_utterances = [
        u for u in utterances
        if len(u.text.strip()) >= min_text_length
    ]

    print(f"Found {len(substantial_utterances)} substantial utterances (>= {min_text_length} characters)")

    if len(substantial_utterances) < num_examples:
        print(f"⚠️  Only {len(substantial_utterances)} substantial utterances available, showing all of them")
        num_examples = len(substantial_utterances)

    # Take sample - sort by length to get most substantial content first
    sample_utterances = sorted(substantial_utterances, key=lambda u: len(u.text), reverse=True)[:num_examples]

    for i, utt in enumerate(sample_utterances, 1):
        text = utt.text.strip()
        topic = utt.meta.get('thread_topic', 'unknown')

        # FIXED: Proper text display with truncation
        if len(text) <= max_text_display:
            display_text = text
            truncated = False
        else:
            # Find the last complete word within the limit
            truncate_at = max_text_display
            while truncate_at > 0 and text[truncate_at] != ' ':
                truncate_at -= 1

            if truncate_at == 0:  # No space found, just cut at the limit
                display_text = text[:max_text_display] + "..."
            else:
                display_text = text[:truncate_at] + "..."
            truncated = True

        # Test different SPINOS-style methods
        simple_result = classify_political_stance_improved(text, method='simple', topic=topic)
        multi_result = classify_political_stance_improved(text, method='multi_template', topic=topic)
        topic_result = classify_with_topic_context(text, topic)

        print(f"\n📋 EXAMPLE {i}/{num_examples}")
        print(f"📂 TOPIC: {topic}")
        print(f"📏 LENGTH: {len(text)} characters")

        if truncated:
            print(f"📄 DISPLAY: Showing first {len(display_text) - 3} characters (truncated)")
        else:
            print(f"📄 DISPLAY: Showing full text ({len(text)} characters)")

        # Show available stance categories for this topic
        available_stances = list(get_stance_hypotheses_for_topic(topic).keys())
        print(f"🎯 STANCE CATEGORIES: {', '.join(available_stances)}")

        print(f"💬 COMMENT TEXT:")
        print("-" * 100)
        print(display_text)
        print("-" * 100)

        print(f"🔍 SPINOS-STYLE STANCE PREDICTIONS:")
        print(f"   Simple Method:      {simple_result['stance']:<20} (conf: {simple_result['confidence']:.3f})")

        multi_consistency_text = f", consistency: {multi_result['template_consistency']:.3f}" if 'template_consistency' in multi_result else ""
        print(
            f"   Multi-Template:     {multi_result['stance']:<20} (conf: {multi_result['confidence']:.3f}{multi_consistency_text})")

        print(f"   Topic-Aware:        {topic_result['stance']:<20} (conf: {topic_result['confidence']:.3f})")

        # Highlight disagreements
        stances = [simple_result['stance'], multi_result['stance'], topic_result['stance']]
        unique_stances = set(stances)

        if len(unique_stances) > 1:
            print("⚠️  🔴 DISAGREEMENT between methods!")
            print(f"   Different predictions: {', '.join(unique_stances)}")
        else:
            print("✅ 🟢 All methods agree!")

        # Show confidence levels
        confidences = [simple_result['confidence'], multi_result['confidence'], topic_result['confidence']]
        avg_confidence = sum(confidences) / len(confidences)
        print(f"📊 Average confidence: {avg_confidence:.3f}")

        if avg_confidence < 0.4:
            print("🔶 LOW CONFIDENCE - may need manual review")
        elif avg_confidence > 0.7:
            print("🔷 HIGH CONFIDENCE - likely reliable prediction")

        print("=" * 140)

    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS for {num_examples} examples:")
    print(
        f"   Average text length: {sum(len(u.text) for u in sample_utterances) / len(sample_utterances):.0f} characters")

    # Topic distribution and stance categories
    topic_counts = {}
    stance_categories_used = set()

    for u in sample_utterances:
        topic = u.meta.get('thread_topic', 'unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Track what stance categories were available
        available_stances = list(get_stance_hypotheses_for_topic(topic).keys())
        stance_categories_used.update(available_stances)

    print(f"   Topic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        available_stances = list(get_stance_hypotheses_for_topic(topic).keys())
        print(f"     • {topic}: {count} utterances (stances: {', '.join(available_stances)})")

    print(f"   All stance categories used: {', '.join(sorted(stance_categories_used))}")


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
                print(f"  {i + 1}. {corpus_dir.name}")

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


def quick_topic_overview(corpus_path):
    """
    Quick function to see what topics are available with their SPINOS stance categories
    """
    print("=== TOPIC OVERVIEW WITH SPINOS STANCE CATEGORIES ===")
    corpus = convokit.Corpus(filename=corpus_path)

    topic_stats = {}
    text_length_stats = {}

    for u in corpus.iter_utterances():
        topic = u.meta.get('thread_topic', 'unknown')
        text_length = len(u.text.strip())

        if topic not in topic_stats:
            topic_stats[topic] = {'count': 0, 'total_length': 0, 'long_posts': 0}

        topic_stats[topic]['count'] += 1
        topic_stats[topic]['total_length'] += text_length

        if text_length >= 200:  # Count substantial posts
            topic_stats[topic]['long_posts'] += 1

    print(f"Found {len(topic_stats)} different topics:")
    print("\nTopic Summary with SPINOS Stance Categories:")
    print("=" * 120)
    print(f"{'Topic':<35} {'Total':<8} {'Long':<8} {'Avg Len':<8} {'% Long':<8} {'SPINOS Stances':<50}")
    print("=" * 120)

    # Sort by number of long posts
    sorted_topics = sorted(topic_stats.items(),
                           key=lambda x: x[1]['long_posts'],
                           reverse=True)

    for topic, stats in sorted_topics:
        avg_length = stats['total_length'] / stats['count'] if stats['count'] > 0 else 0
        percent_long = (stats['long_posts'] / stats['count'] * 100) if stats['count'] > 0 else 0

        # Get SPINOS stance categories for this topic
        available_stances = list(get_stance_hypotheses_for_topic(topic).keys())
        stance_str = ', '.join(available_stances)
        if len(stance_str) > 45:
            stance_str = stance_str[:42] + "..."

        print(
            f"{topic[:34]:<35} {stats['count']:<8} {stats['long_posts']:<8} {avg_length:<8.0f} {percent_long:<8.1f}% {stance_str:<50}")

    print("=" * 120)

    # Enhanced recommendations with SPINOS focus
    print("\n📋 SPINOS RECOMMENDATIONS:")

    # Find topics with specific stance categories
    abortion_topics = [topic for topic, _ in sorted_topics
                       if any(keyword in topic.lower() for keyword in ['abortion', 'reproductive'])]
    gun_topics = [topic for topic, _ in sorted_topics
                  if any(keyword in topic.lower() for keyword in ['gun', 'firearm', 'second amendment'])]

    if abortion_topics:
        abortion_topic = abortion_topics[0]
        abortion_stances = list(get_stance_hypotheses_for_topic(abortion_topic).keys())
        print(f"🎯 Best abortion topic: '{abortion_topic}' (stances: {', '.join(abortion_stances)})")

    if gun_topics:
        gun_topic = gun_topics[0]
        gun_stances = list(get_stance_hypotheses_for_topic(gun_topic).keys())
        print(f"🎯 Best gun topic: '{gun_topic}' (stances: {', '.join(gun_stances)})")

    best_topics = [topic for topic, stats in sorted_topics[:5] if stats['long_posts'] >= 20]
    if best_topics:
        print(f"🎯 Best topics for analysis (most substantial content): {', '.join(best_topics[:3])}")

    total_long_posts = sum(stats['long_posts'] for stats in topic_stats.values())
    print(f"📊 Total substantial posts (>=200 chars): {total_long_posts}")

    # Show stance category variety
    all_stance_categories = set()
    for topic, _ in sorted_topics:
        stances = get_stance_hypotheses_for_topic(topic)
        all_stance_categories.update(stances.keys())

    print(f"📊 Total unique SPINOS stance categories across all topics: {len(all_stance_categories)}")
    print(f"📊 Stance categories: {', '.join(sorted(all_stance_categories))}")

    return topic_stats


def get_topic_specific_examples(corpus_path, topic_name, num_examples=10, min_length=150):
    """
    Get examples from a specific topic for detailed SPINOS-style analysis
    """
    print(f"Loading SPINOS-style examples specifically from topic: {topic_name}")
    corpus = convokit.Corpus(filename=corpus_path)

    # Show what stance categories are available for this topic
    available_stances = list(get_stance_hypotheses_for_topic(topic_name).keys())
    print(f"Available SPINOS stance categories for '{topic_name}': {', '.join(available_stances)}")

    # Find all utterances from this topic
    topic_utterances = [
        u for u in corpus.iter_utterances()
        if u.meta.get('thread_topic') == topic_name and len(u.text.strip()) >= min_length
    ]

    print(f"Found {len(topic_utterances)} substantial utterances in topic '{topic_name}'")

    if len(topic_utterances) == 0:
        print("No utterances found for this topic. Available topics:")
        all_topics = set(u.meta.get('thread_topic', 'unknown') for u in corpus.iter_utterances())
        for topic in sorted(all_topics):
            count = sum(1 for u in corpus.iter_utterances() if u.meta.get('thread_topic') == topic)
            topic_stances = list(get_stance_hypotheses_for_topic(topic).keys())
            print(f"  • {topic}: {count} utterances (stances: {', '.join(topic_stances)})")
        return

    # Sort by length and take the most substantial ones
    sample = sorted(topic_utterances, key=lambda u: len(u.text), reverse=True)[:num_examples]

    preview_stance_classification_comparison(
        corpus_path,
        num_examples=len(sample),
        topic_filter=topic_name,
        min_text_length=min_length,
        max_text_display=1000  # Show more text for topic-specific analysis
    )


def add_stance_metadata_to_corpus(corpus_path, output_path=None, max_utterances=None,
                                  sample_strategy='first', classification_method='multi_template'):
    """
    Enhanced version with SPINOS-style topic-specific stance classification
    """
    # Load the original corpus
    print(f"Loading corpus from {corpus_path}...")
    corpus = convokit.Corpus(filename=corpus_path)

    # Create output path if not provided
    if output_path is None:
        base_name = os.path.basename(corpus_path.rstrip('/'))
        method_suffix = f"_{classification_method}_spinos" if classification_method != 'simple' else "_spinos"
        if max_utterances:
            output_path = f"{base_name}_with_stances{method_suffix}_subset_{max_utterances}"
        else:
            output_path = f"{base_name}_with_stances{method_suffix}"

    print(f"Will save SPINOS-style stance-annotated corpus to: {output_path}")
    print(f"Using classification method: {classification_method}")
    print(f"Using topic-specific stances (pro-abortion, anti-gun, etc.) instead of political labels")

    # Get all utterances and apply sampling strategy
    all_utterances = list(corpus.iter_utterances())
    total_available = len(all_utterances)

    if max_utterances is None or max_utterances >= total_available:
        utterances_to_process = all_utterances
        print(f"Processing all {total_available} utterances...")
    else:
        print(
            f"Selecting {max_utterances} utterances from {total_available} total using '{sample_strategy}' strategy...")

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
                topic_count = per_topic + (1 if i < remainder else 0)
                topic_sample = topic_utterances[:min(topic_count, len(topic_utterances))]
                utterances_to_process.extend(topic_sample)
                print(f"  {topic}: {len(topic_sample)} utterances")
        else:
            raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

    # Process selected utterances with chosen method
    print(
        f"Classifying SPINOS-style political stances for {len(utterances_to_process)} utterances using {classification_method}...")

    stance_counts = {}
    topic_stance_distribution = {}
    processed_count = 0
    method_stats = {
        'high_confidence': 0,
        'low_confidence': 0,
        'template_consistency_scores': [],
        'topic_specific_classifications': 0
    }

    for utterance in tqdm(utterances_to_process, desc="Processing utterances"):
        text = utterance.text
        topic = utterance.meta.get('thread_topic', 'unknown')

        # Skip very short texts
        if len(text.strip()) < 10:
            utterance.meta['political_stance'] = 'neutral'
            utterance.meta['stance_confidence'] = 0.0
            utterance.meta['stance_method'] = classification_method + '_spinos'
            utterance.meta['topic_context'] = topic
            continue

        # Classify stance using chosen method with SPINOS-style categories
        if classification_method == 'topic_aware':
            stance_result = classify_with_topic_context(text, topic)
        else:
            stance_result = classify_political_stance_improved(text, method=classification_method, topic=topic)

        # Add metadata to utterance
        utterance.meta['political_stance'] = stance_result['stance']
        utterance.meta['stance_confidence'] = stance_result['confidence']
        utterance.meta['stance_method'] = stance_result['method_used']
        utterance.meta['stance_reliable'] = stance_result.get('reliable', False)
        utterance.meta['topic_context'] = stance_result.get('topic_context', topic)

        if 'template_consistency' in stance_result:
            utterance.meta['template_consistency'] = stance_result['template_consistency']
            method_stats['template_consistency_scores'].append(stance_result['template_consistency'])

        # Track statistics
        stance = stance_result['stance']
        stance_counts[stance] = stance_counts.get(stance, 0) + 1

        # Track topic-specific stance distribution
        if topic not in topic_stance_distribution:
            topic_stance_distribution[topic] = {}
        topic_stance_distribution[topic][stance] = topic_stance_distribution[topic].get(stance, 0) + 1

        if stance_result['confidence'] > 0.5:
            method_stats['high_confidence'] += 1
        else:
            method_stats['low_confidence'] += 1

        # Count topic-specific classifications
        if stance_result.get('topic_context') and stance_result['topic_context'] != 'unknown':
            method_stats['topic_specific_classifications'] += 1

        processed_count += 1

        # Progress update
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(utterances_to_process)} utterances...")

    # Save corpus and statistics
    if max_utterances and max_utterances < total_available:
        print("Creating subset corpus...")
        subset_corpus = convokit.Corpus(utterances=utterances_to_process)
        print(f"Saving subset corpus to {output_path}...")
        subset_corpus.dump(output_path)
        final_corpus = subset_corpus
    else:
        print(f"Saving full annotated corpus to {output_path}...")
        corpus.dump(output_path)
        final_corpus = corpus

    # Enhanced statistics
    avg_consistency = np.mean(method_stats['template_consistency_scores']) if method_stats[
        'template_consistency_scores'] else 0.0

    try:
        stats_file = os.path.expanduser(f"~/.convokit/saved-corpora/{output_path}/stance_statistics.json")
        stats = {
            'total_utterances_in_corpus': total_available,
            'processed_utterances': processed_count,
            'sampling_strategy': sample_strategy,
            'classification_method': classification_method + '_spinos',
            'max_utterances_requested': max_utterances,
            'stance_distribution': stance_counts,
            'topic_stance_distribution': topic_stance_distribution,
            'method_performance': {
                'high_confidence_predictions': method_stats['high_confidence'],
                'low_confidence_predictions': method_stats['low_confidence'],
                'average_template_consistency': avg_consistency,
                'confidence_rate': method_stats['high_confidence'] / processed_count if processed_count > 0 else 0,
                'topic_specific_rate': method_stats[
                                           'topic_specific_classifications'] / processed_count if processed_count > 0 else 0
            },
            'stance_categories_by_topic': {
                topic: list(get_stance_hypotheses_for_topic(topic).keys())
                for topic in set(u.meta.get('thread_topic', 'unknown') for u in utterances_to_process)
            }
        }

        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("✅ Enhanced SPINOS-style stance statistics saved successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not save statistics: {e}")

    print("Enhanced SPINOS-style stance classification completed!")
    print(f"Overall stance distribution: {stance_counts}")
    print(f"Average template consistency: {avg_consistency:.3f}")
    print(f"High confidence rate: {method_stats['high_confidence'] / processed_count:.1%}")
    print(f"Topic-specific classification rate: {method_stats['topic_specific_classifications'] / processed_count:.1%}")
    print(f"Final output location: ~/.convokit/saved-corpora/{output_path}")

    return final_corpus


def analyze_stance_distribution_by_topic(corpus_path, method='multi_template', sample_size=100):
    """
    Analyze how SPINOS-style stance predictions vary by topic
    """
    print(f"=== SPINOS-STYLE STANCE DISTRIBUTION BY TOPIC ANALYSIS ===")
    print(f"Using method: {method}, sample size: {sample_size} per topic")
    print(f"Topic-specific stances: pro-abortion vs anti-abortion (not liberal vs conservative)")

    corpus = convokit.Corpus(filename=corpus_path)

    # Group utterances by topic
    topic_groups = {}
    for u in corpus.iter_utterances():
        topic = u.meta.get('thread_topic', 'unknown')
        if len(u.text.strip()) >= 100:  # Only substantial posts
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(u)

    # Analyze top topics
    topic_results = {}
    for topic, utterances in topic_groups.items():
        if len(utterances) < 20:  # Skip topics with too few posts
            continue

        # Sample utterances
        sample = utterances[:min(sample_size, len(utterances))]

        stance_counts = {}
        total_confidence = 0

        # Get available stance categories for this topic
        available_stances = list(get_stance_hypotheses_for_topic(topic).keys())

        for utt in sample:
            if method == 'topic_aware':
                result = classify_with_topic_context(utt.text, topic)
            else:
                result = classify_political_stance_improved(utt.text, method=method, topic=topic)

            stance = result['stance']
            stance_counts[stance] = stance_counts.get(stance, 0) + 1
            total_confidence += result['confidence']

        topic_results[topic] = {
            'sample_size': len(sample),
            'stance_distribution': stance_counts,
            'avg_confidence': total_confidence / len(sample),
            'total_available': len(utterances),
            'available_stances': available_stances
        }

    # Display results
    print(f"\nAnalyzed {len(topic_results)} topics:")
    print("=" * 120)

    for topic, results in sorted(topic_results.items(),
                                 key=lambda x: x[1]['avg_confidence'],
                                 reverse=True):
        print(f"\n🏷️  TOPIC: {topic}")
        print(f"   Sample: {results['sample_size']}/{results['total_available']} posts")
        print(f"   Avg Confidence: {results['avg_confidence']:.3f}")
        print(f"   Available Stances: {', '.join(results['available_stances'])}")
        print(f"   Stance Distribution:")

        total_predictions = sum(results['stance_distribution'].values())
        for stance, count in sorted(results['stance_distribution'].items(),
                                    key=lambda x: x[1],
                                    reverse=True):
            percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
            print(f"     • {stance}: {count} ({percentage:.1f}%)")

    return topic_results


# Main execution with SPINOS integration
if __name__ == "__main__":
    # Try to automatically find your processed corpus
    print("=== FINDING PROCESSED CORPUS ===")
    CORPUS_PATH = find_processed_corpus()

    if not CORPUS_PATH:
        # Fallback: manual entry
        print("Could not automatically find processed corpus.")
        CORPUS_PATH = input("Please enter the corpus name: ")
        if not CORPUS_PATH.startswith('~') and not os.path.isabs(CORPUS_PATH):
            CORPUS_PATH = os.path.expanduser(f"~/.convokit/saved-corpora/{CORPUS_PATH}")

    print(f"Using corpus: {CORPUS_PATH}")

    # Enhanced preview options with SPINOS focus
    print("\n=== SPINOS-STYLE STANCE CLASSIFICATION PREVIEW ===")
    print("🎯 Now using topic-specific stances (pro-abortion, anti-gun, etc.) instead of political labels!")
    print("1. Enhanced general preview (longer posts, SPINOS stances)")
    print("2. Topic-specific analysis (deep dive with topic stances)")
    print("3. Quick method comparison (SPINOS vs original)")

    preview_choice = input("Choose preview option (1-3): ")

    if preview_choice == '1':
        print("\n=== ENHANCED SPINOS-STYLE GENERAL PREVIEW ===")
        num_examples = int(input("Number of examples to show (default 10): ") or "10")
        min_length = int(input("Minimum text length in characters (default 150): ") or "150")
        max_display = int(input("Maximum characters to display per text (default 800): ") or "800")

        preview_stance_classification_comparison(
            CORPUS_PATH,
            num_examples=num_examples,
            min_text_length=min_length,
            max_text_display=max_display
        )

    elif preview_choice == '2':
        print("\n=== TOPIC-SPECIFIC SPINOS ANALYSIS ===")
        # First, show available topics with their stance categories
        corpus = convokit.Corpus(filename=CORPUS_PATH)
        all_topics = {}
        for u in corpus.iter_utterances():
            topic = u.meta.get('thread_topic', 'unknown')
            all_topics[topic] = all_topics.get(topic, 0) + 1

        print("Available topics with their SPINOS stance categories:")
        sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)
        for i, (topic, count) in enumerate(sorted_topics[:15]):  # Show top 15
            available_stances = list(get_stance_hypotheses_for_topic(topic).keys())
            print(f"  {i + 1}. {topic} ({count} utterances)")
            print(f"      Stances: {', '.join(available_stances)}")

        topic_choice = input("\nEnter topic name (or number): ")

        # Handle numeric choice
        try:
            topic_idx = int(topic_choice) - 1
            if 0 <= topic_idx < len(sorted_topics):
                selected_topic = sorted_topics[topic_idx][0]
            else:
                selected_topic = topic_choice
        except ValueError:
            selected_topic = topic_choice

        num_examples = int(input("Number of examples (default 8): ") or "8")
        min_length = int(input("Minimum text length (default 200): ") or "200")

        get_topic_specific_examples(
            CORPUS_PATH,
            selected_topic,
            num_examples=num_examples,
            min_length=min_length
        )

    elif preview_choice == '3':
        print("\n=== QUICK SPINOS METHOD COMPARISON ===")
        # Compare SPINOS vs original approach
        preview_stance_classification_comparison(CORPUS_PATH, num_examples=10)

    # Ask user about method preference
    print("\n=== SPINOS-STYLE CLASSIFICATION METHOD SELECTION ===")
    print("All methods now use topic-specific stances (pro-abortion vs anti-abortion, etc.)")
    print("1. Simple SPINOS (topic-aware stance categories)")
    print("2. Multi-template SPINOS (research-optimized with topic stances)")
    print("3. Topic-aware SPINOS (full context enhancement)")

    method_choice = input("Choose classification method (1-3): ")
    method_map = {
        '1': 'simple',
        '2': 'multi_template',
        '3': 'topic_aware'
    }

    selected_method = method_map.get(method_choice, 'multi_template')
    print(f"Selected method: {selected_method} (SPINOS-style)")

    # Ask user if they want to continue
    response = input("\nDoes one of the SPINOS methods look better? Continue with processing? (y/n): ")

    if response.lower() == 'y':
        # Processing options
        print("\n=== SPINOS PROCESSING OPTIONS ===")
        print("0. Process tiny subset (50 utterances) - for quick SPINOS method testing")
        print("1. Process small subset (500 utterances) - for SPINOS method validation")
        print("2. Process medium subset (2000 utterances) - for SPINOS development")
        print("3. Process large subset (10000 utterances) - for full SPINOS analysis")
        print("4. Process everything - full SPINOS dataset")

        choice = input("Choose option (0-4): ")

        if choice == '0':
            max_utterances = 50
            strategy = 'balanced_topics'
        elif choice == '1':
            max_utterances = 500
            strategy = 'balanced_topics'
        elif choice == '2':
            max_utterances = 2000
            strategy = 'balanced_topics'
        elif choice == '3':
            max_utterances = 10000
            strategy = 'random'
        elif choice == '4':
            max_utterances = None
            strategy = 'first'
        else:
            print("Invalid choice. Using default.")
            max_utterances = 500
            strategy = 'balanced_topics'

        print(f"\n=== PROCESSING {max_utterances or 'ALL'} UTTERANCES WITH SPINOS ===")
        annotated_corpus = add_stance_metadata_to_corpus(
            CORPUS_PATH,
            max_utterances=max_utterances,
            sample_strategy=strategy,
            classification_method=selected_method
        )

        print("Done! Check the SPINOS results and statistics.")

    else:
        print("Stopping here. You can adjust the SPINOS methods or test different approaches.")

    # Add new option for SPINOS topic analysis
    print("\n=== ADDITIONAL SPINOS ANALYSIS OPTIONS ===")
    print("A. Quick topic overview (with SPINOS stance categories)")
    print("B. SPINOS stance distribution by topic")
    print("C. Skip additional analysis")

    analysis_choice = input("Choose additional analysis (A/B/C): ").upper()

    if analysis_choice == 'A':
        topic_stats = quick_topic_overview(CORPUS_PATH)
        print("\n🎯 SPINOS Stance Categories by Topic:")
        for topic in sorted(topic_stats.keys()):
            if topic_stats[topic]['long_posts'] >= 10:  # Only show topics with sufficient content
                available_stances = list(get_stance_hypotheses_for_topic(topic).keys())
                print(f"  • {topic}: {', '.join(available_stances)}")

    elif analysis_choice == 'B':
        method_for_analysis = selected_method if 'selected_method' in locals() else 'multi_template'
        analyze_stance_distribution_by_topic(CORPUS_PATH, method=method_for_analysis)

    print("\n✅ SPINOS-style analysis complete!")