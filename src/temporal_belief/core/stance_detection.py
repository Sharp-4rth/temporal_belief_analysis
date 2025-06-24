"""Stance detection functionality for political text analysis."""

from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm

from ..models.bart_classifier import BARTZeroShotClassifier
from ..utils.config import STANCE_LABELS

logger = logging.getLogger(__name__)

class StanceDetector:
    """Detect political stance in ConvoKit utterances using BART."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli",
                 stance_labels: Optional[List[str]] = None):
        """Initialize stance detector."""
        self.classifier = BARTZeroShotClassifier(model_name)
        self.stance_labels = stance_labels or STANCE_LABELS
        logger.info(f"Initialized stance detector with labels: {self.stance_labels}")

    def detect_utterance_stance(self, utterance) -> Dict[str, Any]:
        """Detect stance for a single utterance."""
        text = utterance.text
        result = self.classifier.classify_text(text, self.stance_labels)

        return {
            "stance": result["label"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"]
        }

    def process_corpus_utterances(self, corpus, batch_size: int = 50) -> None:
        """Process all utterances in corpus for stance detection."""
        utterances = list(corpus.iter_utterances())
        logger.info(f"Processing {len(utterances)} utterances for stance detection")

        for i in tqdm(range(0, len(utterances), batch_size),
                      desc="Processing utterances"):
            batch = utterances[i:i+batch_size]

            for utt in batch:
                try:
                    stance_result = self.detect_utterance_stance(utt)

                    # Add to utterance metadata
                    utt.add_meta("detected_stance", stance_result["stance"])
                    utt.add_meta("stance_confidence", stance_result["confidence"])
                    utt.add_meta("stance_scores", stance_result["all_scores"])

                except Exception as e:
                    logger.error(f"Failed to process utterance {utt.id}: {e}")
                    utt.add_meta("detected_stance", "unknown")
                    utt.add_meta("stance_confidence", 0.0)