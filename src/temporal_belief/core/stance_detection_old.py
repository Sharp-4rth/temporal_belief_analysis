"""Stance detection functionality for political text analysis."""

from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm

from ..models.bart_classifier import BARTZeroShotClassifier
from ..utils.config import TOPIC_STANCE_HYPOTHESES, POLITICAL_TOPICS, ProjectConfig

logger = logging.getLogger(__name__)

class StanceDetector:
    """Detect political stance in ConvoKit utterances using BART."""

    def __init__(self, stance_labels: Optional[List[str]] = None,
                 config: ProjectConfig = None):
        """Initialize stance detector."""
        self.config = config or ProjectConfig()
        self.classifier = BARTZeroShotClassifier(self.config.bart_model_name)
        self.stance_labels = stance_labels or TOPIC_STANCE_HYPOTHESES
        logger.info(f"Initialized stance detector with labels: {self.stance_labels}")


    def detect_utterance_stance(self, utterance, corpus) -> Dict[str, Any]:
        """Detect stance for a single utterance."""
        # text = mark_quotes(utterance.text)
        if not utterance.text or utterance.text == '[removed]' or utterance.text == '[deleted]' or utterance.text.strip() == '.':
            logger.warning(f"No utterance found in {utterance.id}")
            return {"stance": "unknown", "confidence": 0.0}
        convo = corpus.get_conversation(utterance.conversation_id)
        topic = convo.meta['detected_topic']
        clean_text = prepare_text(utterance.text)
        text = get_contextual_framing_for_topic(topic, clean_text)
        # MAKE SURE THE TOPIC KEYS MATCH AND IF NOT RETURN A GENERAL
        stance_hypotheses = TOPIC_STANCE_HYPOTHESES.get(topic, GENERAL_STANCE_HYPOTHESES)
        logger.info(f"Text: {text}")
        #
        # # Pass one of the candidate stances
        # result = self.classifier.classify_text(text, self.stance_labels)
        # return {
        #     "stance": result["label"],
        #     "confidence": result["confidence"],
        #     "all_scores": result["all_scores"]
        # }

        confidence_threshold = 0.25
        stance_results = {}
        template_consistency_scores = {}

        for stance, hypotheses in stance_hypotheses.items():
            stance_scores = []

            # Test each hypothesis template for this stance
            for hypothesis in hypotheses:
                result = self.classifier.classify_text(text, [hypothesis])
                # Get score for this specific hypothesis
                stance_scores.append(result["all_scores"].get(hypothesis, 0.0))

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

    def process_corpus_utterances(self, corpus, batch_size: int = 50,
                                  max_utterances: Optional[int] = None,
                              save_path: Optional[str] = None) -> None:
        """Process all utterances in corpus for stance detection."""
        sorted_utts = sorted(list(corpus.iter_utterances()), key=lambda utt: utt.timestamp)
        all_utterances = sorted_utts

        if max_utterances is not None:
            utterances = all_utterances[:max_utterances]
            logger.info(f"Processing {len(utterances)} of {len(all_utterances)} total utterances")
        else:
            utterances = all_utterances
            logger.info(f"Processing all {len(utterances)} utterances for stance detection")

        for i in tqdm(range(0, len(utterances), batch_size),
                      desc="Processing utterances"):
            batch = utterances[i:i+batch_size]

            for utt in batch:
                try:
                    stance_result = self.detect_utterance_stance(utt, corpus)

                    # Add to utterance metadata
                    utt.add_meta("detected_stance", stance_result["stance"])
                    utt.add_meta("stance_confidence", stance_result["confidence"])
                    utt.add_meta("stance_scores", stance_result["all_scores"])

                except Exception as e:
                    logger.error(f"Failed to process utterance {utt.id}: {e}")
                    utt.add_meta("detected_stance", "unknown")
                    utt.add_meta("stance_confidence", 0.0)

        if save_path:
            corpus.dump(save_path)
            logger.info(f"Saved processed corpus to {save_path}")

        logger.info("Stance detection processing complete")