"""BART-based zero-shot classification for topic and stance detection."""

from transformers import pipeline
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BARTZeroShotClassifier:
    """Generic BART-based zero-shot classifier."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize BART classifier.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.classifier = None
        self._load_model()

    def _load_model(self):
        """Load the BART model for zero-shot classification."""
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name
            )
            logger.info(f"Loaded BART model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load BART model: {e}")
            raise

    def classify_text(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Classify text into one of the candidate labels.

        Args:
            text: Input text to classify
            candidate_labels: List of possible labels

        Returns:
            Dictionary with top label, confidence, and all scores
        """
        if not self.classifier:
            raise RuntimeError("Classifier not initialized")

        if not text.strip():
            return {
                "label": "unknown",
                "confidence": 0.0,
                "all_scores": {label: 0.0 for label in candidate_labels}
            }

        try:
            result = self.classifier(text, candidate_labels)

            return {
                "label": result["labels"][0],
                "confidence": result["scores"][0],
                "all_scores": dict(zip(result["labels"], result["scores"]))
            }
        except Exception as e:
            logger.error(f"Classification failed for text: {text[:100]}... Error: {e}")
            return {
                "label": "unknown",
                "confidence": 0.0,
                "all_scores": {label: 0.0 for label in candidate_labels}
            }