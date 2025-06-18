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

# RESEARCH-BASED IMPROVED HYPOTHESIS TEMPLATES
# Based on findings that specific, action-oriented hypotheses perform better
STANCE_HYPOTHESES = {
    'strongly liberal/progressive': [
        'The author strongly supports progressive political policies and liberal values',
        'This comment advocates for far-left political positions and social justice',
        'The author expresses strong liberal activist viewpoints and reform ideas'
    ],
    'moderately liberal/left-leaning': [
        'The author moderately supports liberal Democratic policies and positions',
        'This comment leans toward center-left political views and reforms',
        'The author expresses mild support for progressive ideas and changes'
    ],
    'centrist/moderate': [
        'The author takes a balanced centrist political position on issues',
        'This comment expresses moderate views without strong partisan preferences',
        'The author seeks pragmatic middle-ground political solutions'
    ],
    'moderately conservative/right-leaning': [
        'The author moderately supports conservative Republican policies and traditions',
        'This comment leans toward center-right political positions and values',
        'The author expresses mild support for traditional conservative approaches'
    ],
    'strongly conservative': [
        'The author strongly supports traditional conservative policies and values',
        'This comment advocates for far-right political positions and traditions',
        'The author expresses strong support for conservative principles and order'
    ],
    'libertarian/anti-government': [
        'The author supports libertarian principles and opposes government control',
        'This comment advocates for individual freedom over government intervention',
        'The author favors minimal government and maximum personal liberty'
    ],
    'authoritarian/pro-strong-government': [
        'The author supports strong governmental authority and centralized control',
        'This comment advocates for increased government power and oversight',
        'The author favors authoritarian approaches to policy and governance'
    ],
    'neutral/informational': [
        'This comment provides neutral factual information without political bias',
        'The author presents objective information without taking political sides',
        'This text is politically neutral and purely informational in nature'
    ]
}

# Alternative simplified hypotheses for comparison
SIMPLE_HYPOTHESES = [
    'strongly liberal/progressive',
    'moderately liberal/left-leaning',
    'centrist/moderate',
    'moderately conservative/right-leaning',
    'strongly conservative',
    'libertarian/anti-government',
    'authoritarian/pro-strong-government',
    'neutral/informational'
]


def classify_political_stance_improved(text, confidence_threshold=0.25, method='multi_template'):
    """
    IMPROVED stance classification using research-based optimizations:
    1. Multiple hypothesis templates per category
    2. Context-aware prompting
    3. Sensitivity analysis across templates
    4. Option for model ensemble

    Args:
        text (str): The comment text to classify
        confidence_threshold (float): Minimum confidence to assign a stance
        method (str): 'multi_template', 'simple', or 'ensemble'

    Returns:
        dict: Enhanced stance classification results
    """
    if not text or len(text.strip()) < 10:
        return {
            'stance': 'neutral/informational',
            'confidence': 0.0,
            'all_scores': {},
            'method_used': method,
            'template_consistency': 0.0
        }

    # Clean and prepare text with context
    clean_text = text.strip()
    if len(clean_text) > 500:  # Limit for better performance
        clean_text = clean_text[:500] + "..."

    # Add contextual framing (research shows this helps)
    contextual_text = f"In a political discussion, this comment states: {clean_text}"

    try:
        if method == 'multi_template':
            return _classify_multi_template(contextual_text, confidence_threshold)
        elif method == 'simple':
            return _classify_simple(contextual_text, confidence_threshold)
        elif method == 'ensemble':
            return _classify_ensemble(contextual_text, confidence_threshold)
        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        print(f"Error classifying stance for text: {str(e)}")
        return {
            'stance': 'error',
            'confidence': 0.0,
            'all_scores': {},
            'method_used': method,
            'template_consistency': 0.0
        }


def _classify_multi_template(text, confidence_threshold):
    """Use multiple hypothesis templates and average results"""
    stance_results = {}
    template_consistency_scores = {}

    for stance, hypotheses in STANCE_HYPOTHESES.items():
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
        best_stance = 'neutral/informational'
        best_confidence = stance_results.get('neutral/informational', 0.0)

    return {
        'stance': best_stance,
        'confidence': best_confidence,
        'all_scores': stance_results,
        'method_used': 'multi_template',
        'template_consistency': overall_consistency,
        'reliable': best_confidence > confidence_threshold and overall_consistency > 0.7
    }


def _classify_simple(text, confidence_threshold):
    """Use simple hypothesis templates (your original approach)"""
    result = bart_classifier(text, SIMPLE_HYPOTHESES)
    top_confidence = result['scores'][0]

    if top_confidence < confidence_threshold:
        stance = 'neutral/informational'
    else:
        stance = result['labels'][0]

    return {
        'stance': stance,
        'confidence': top_confidence,
        'all_scores': dict(zip(result['labels'], result['scores'])),
        'method_used': 'simple',
        'template_consistency': 1.0,  # N/A for simple method
        'reliable': top_confidence > confidence_threshold
    }


def _classify_ensemble(text, confidence_threshold):
    """Use ensemble of BART + DeBERTa (requires deberta_classifier to be uncommented)"""
    # Note: This requires uncommenting the DeBERTa model above
    try:
        # Get results from both models
        bart_result = bart_classifier(text, SIMPLE_HYPOTHESES)
        # deberta_result = deberta_classifier(text, SIMPLE_HYPOTHESES)

        # For now, just use BART (uncomment above line to use ensemble)
        bart_probs = np.array(bart_result['scores'])
        # deberta_probs = np.array(deberta_result['scores'])

        # Weighted ensemble (research shows DeBERTa slightly better)
        # ensemble_probs = 0.4 * bart_probs + 0.6 * deberta_probs
        ensemble_probs = bart_probs  # Fallback to BART only

        best_idx = np.argmax(ensemble_probs)
        best_stance = SIMPLE_HYPOTHESES[best_idx]
        best_confidence = ensemble_probs[best_idx]

        if best_confidence < confidence_threshold:
            best_stance = 'neutral/informational'

        return {
            'stance': best_stance,
            'confidence': best_confidence,
            'all_scores': dict(zip(SIMPLE_HYPOTHESES, ensemble_probs)),
            'method_used': 'ensemble',
            'template_consistency': 1.0,  # N/A for ensemble
            'reliable': best_confidence > confidence_threshold
        }

    except Exception as e:
        print(f"Ensemble failed, falling back to BART: {e}")
        return _classify_simple(text, confidence_threshold)


def classify_with_topic_context(text, topic=None, confidence_threshold=0.25):
    """
    Enhanced classification that uses topic-specific context and hypotheses
    """
    if not text or len(text.strip()) < 10:
        return classify_political_stance_improved(text, confidence_threshold)

    # Topic-specific contextual framing
    if topic and topic != 'general_discussion':
        contextual_text = f"In a discussion about {topic}, this comment states: {text.strip()}"

        # Topic-specific hypothesis adjustments
        if topic in ['abortion and reproductive rights']:
            topic_hypotheses = {
                'strongly liberal/progressive': [
                    'The author strongly supports pro-choice reproductive rights',
                    'This comment advocates for abortion access and reproductive freedom'
                ],
                'strongly conservative': [
                    'The author strongly supports pro-life anti-abortion positions',
                    'This comment advocates for protecting unborn life and traditional values'
                ],
                'neutral/informational': [
                    'This comment provides factual information about reproductive policy',
                    'The author presents neutral information about abortion laws'
                ]
            }
        elif topic in ['gun rights and control']:
            topic_hypotheses = {
                'strongly liberal/progressive': [
                    'The author strongly supports gun control and firearm restrictions',
                    'This comment advocates for stronger gun safety measures'
                ],
                'strongly conservative': [
                    'The author strongly supports gun rights and Second Amendment freedoms',
                    'This comment advocates for protecting firearm ownership rights'
                ],
                'neutral/informational': [
                    'This comment provides factual information about gun policy',
                    'The author presents neutral information about firearm laws'
                ]
            }
        else:
            # Use general hypotheses for other topics
            topic_hypotheses = STANCE_HYPOTHESES
    else:
        # No topic context available
        contextual_text = f"In a political discussion, this comment states: {text.strip()}"
        topic_hypotheses = STANCE_HYPOTHESES

    # Use the enhanced classification with topic context
    result = _classify_multi_template_custom(contextual_text, topic_hypotheses, confidence_threshold)
    result['topic_context'] = topic
    return result


def _classify_multi_template_custom(text, hypotheses_dict, confidence_threshold):
    """Multi-template classification with custom hypotheses"""
    stance_results = {}

    for stance, hypotheses in hypotheses_dict.items():
        if hypotheses:  # Only process if hypotheses exist for this stance
            stance_scores = []
            for hypothesis in hypotheses:
                result = bart_classifier(text, [hypothesis])
                stance_scores.append(result['scores'][0] if result['labels'][0] == hypothesis else 0)
            stance_results[stance] = np.mean(stance_scores)
        else:
            stance_results[stance] = 0.0

    # Find best stance
    best_stance = max(stance_results.keys(), key=lambda k: stance_results[k])
    best_confidence = stance_results[best_stance]

    if best_confidence < confidence_threshold:
        best_stance = 'neutral/informational'
        best_confidence = stance_results.get('neutral/informational', 0.0)

    return {
        'stance': best_stance,
        'confidence': best_confidence,
        'all_scores': stance_results,
        'method_used': 'topic_aware_multi_template',
        'reliable': best_confidence > confidence_threshold
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


def preview_stance_classification_comparison(corpus_path, num_examples=20, topic_filter=None):
    """
    Compare different stance classification methods side by side
    """
    print("Loading corpus for stance classification method comparison...")
    corpus = convokit.Corpus(filename=corpus_path)

    print(f"\nComparing stance classification methods on {num_examples} comments:")
    if topic_filter:
        print(f"Filtering for topic: {topic_filter}")
    print("-" * 120)

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

        # Test different methods
        simple_result = classify_political_stance_improved(text, method='simple')
        multi_result = classify_political_stance_improved(text, method='multi_template')
        topic_result = classify_with_topic_context(text, topic)

        print(f"Topic: {topic}")
        print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"Simple Method:      {simple_result['stance']:<35} (conf: {simple_result['confidence']:.3f})")
        print(
            f"Multi-Template:     {multi_result['stance']:<35} (conf: {multi_result['confidence']:.3f}, consistency: {multi_result['template_consistency']:.3f})")
        print(f"Topic-Aware:        {topic_result['stance']:<35} (conf: {topic_result['confidence']:.3f})")

        # Highlight disagreements
        stances = [simple_result['stance'], multi_result['stance'], topic_result['stance']]
        if len(set(stances)) > 1:
            print("⚠️  DISAGREEMENT between methods!")

        print("-" * 120)


def add_stance_metadata_to_corpus(corpus_path, output_path=None, max_utterances=None,
                                  sample_strategy='first', classification_method='multi_template'):
    """
    Enhanced version with improved stance classification methods
    """
    # Load the original corpus
    print(f"Loading corpus from {corpus_path}...")
    corpus = convokit.Corpus(filename=corpus_path)

    # Create output path if not provided
    if output_path is None:
        base_name = os.path.basename(corpus_path.rstrip('/'))
        method_suffix = f"_{classification_method}" if classification_method != 'simple' else ""
        if max_utterances:
            output_path = f"{base_name}_with_stances{method_suffix}_subset_{max_utterances}"
        else:
            output_path = f"{base_name}_with_stances{method_suffix}"

    print(f"Will save stance-annotated corpus to: {output_path}")
    print(f"Using classification method: {classification_method}")

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
    print(f"Classifying political stances for {len(utterances_to_process)} utterances using {classification_method}...")

    stance_counts = {}
    processed_count = 0
    method_stats = {
        'high_confidence': 0,
        'low_confidence': 0,
        'template_consistency_scores': [],
        'disagreements_with_simple': 0
    }

    for utterance in tqdm(utterances_to_process, desc="Processing utterances"):
        text = utterance.text
        topic = utterance.meta.get('thread_topic', 'unknown')

        # Skip very short texts
        if len(text.strip()) < 10:
            utterance.meta['political_stance'] = 'neutral/informational'
            utterance.meta['stance_confidence'] = 0.0
            utterance.meta['stance_method'] = classification_method
            continue

        # Classify stance using chosen method
        if classification_method == 'topic_aware':
            stance_result = classify_with_topic_context(text, topic)
        else:
            stance_result = classify_political_stance_improved(text, method=classification_method)

        # Add metadata to utterance
        utterance.meta['political_stance'] = stance_result['stance']
        utterance.meta['stance_confidence'] = stance_result['confidence']
        utterance.meta['stance_method'] = stance_result['method_used']
        utterance.meta['stance_reliable'] = stance_result.get('reliable', False)

        if 'template_consistency' in stance_result:
            utterance.meta['template_consistency'] = stance_result['template_consistency']
            method_stats['template_consistency_scores'].append(stance_result['template_consistency'])

        # Track statistics
        stance = stance_result['stance']
        stance_counts[stance] = stance_counts.get(stance, 0) + 1

        if stance_result['confidence'] > 0.5:
            method_stats['high_confidence'] += 1
        else:
            method_stats['low_confidence'] += 1

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
            'classification_method': classification_method,
            'max_utterances_requested': max_utterances,
            'stance_distribution': stance_counts,
            'method_performance': {
                'high_confidence_predictions': method_stats['high_confidence'],
                'low_confidence_predictions': method_stats['low_confidence'],
                'average_template_consistency': avg_consistency,
                'confidence_rate': method_stats['high_confidence'] / processed_count if processed_count > 0 else 0
            },
            'stance_categories': list(STANCE_HYPOTHESES.keys())
        }

        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("✅ Enhanced stance statistics saved successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not save statistics: {e}")

    print("Enhanced stance classification completed!")
    print(f"Stance distribution: {stance_counts}")
    print(f"Average template consistency: {avg_consistency:.3f}")
    print(f"High confidence rate: {method_stats['high_confidence'] / processed_count:.1%}")
    print(f"Final output location: ~/.convokit/saved-corpora/{output_path}")

    return final_corpus


# Main execution
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

    # Preview with method comparison
    print("\n=== STANCE CLASSIFICATION METHOD COMPARISON ===")
    preview_stance_classification_comparison(CORPUS_PATH, num_examples=10)

    # Ask user about method preference
    print("\n=== CLASSIFICATION METHOD SELECTION ===")
    print("1. Simple (original approach)")
    print("2. Multi-template (research-optimized)")
    print("3. Topic-aware (context-enhanced)")

    method_choice = input("Choose classification method (1-3): ")
    method_map = {
        '1': 'simple',
        '2': 'multi_template',
        '3': 'topic_aware'
    }

    selected_method = method_map.get(method_choice, 'multi_template')
    print(f"Selected method: {selected_method}")

    # Ask user if they want to continue
    response = input("\nDoes one of the methods look better? Continue with processing? (y/n): ")

    if response.lower() == 'y':
        # Processing options
        print("\n=== PROCESSING OPTIONS ===")
        print("0. Process tiny subset (50 utterances) - for quick method testing")
        print("1. Process small subset (500 utterances) - for method validation")
        print("2. Process medium subset (2000 utterances) - for development")
        print("3. Process large subset (10000 utterances) - for full analysis")
        print("4. Process everything - full dataset")

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

        print(f"\n=== PROCESSING {max_utterances or 'ALL'} UTTERANCES ===")
        annotated_corpus = add_stance_metadata_to_corpus(
            CORPUS_PATH,
            max_utterances=max_utterances,
            sample_strategy=strategy,
            classification_method=selected_method
        )

        print("Done! Check the results and statistics.")

    else:
        print("Stopping here. You can adjust the methods or test different approaches.")