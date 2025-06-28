"""BART-based zero-shot classification for topic and stance detection."""

from transformers import pipeline
from typing import List, Dict, Any, Optional
import logging
import torch

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
            # Add device parameter for GPU acceleration
            device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else -1

            self.classifier = pipeline(
                task="zero-shot-classification",
                model=self.model_name,
                device=device
            )
            logger.info(f"Loaded BART model: {self.model_name} on device: {'GPU' if device == 0 else 'CPU'}")
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
            # Pass the text and candidate labels
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

    def classify_batch(self, texts: List[str], candidate_labels: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts at once for better GPU utilization.

        Args:
            texts: List of input texts to classify
            candidate_labels: List of possible labels

        Returns:
            List of classification results, one per input text
        """
        if not self.classifier:
            raise RuntimeError("Classifier not initialized")

        if not texts:
            return []

        try:
            # Process all texts at once - this is where the speedup happens
            results = self.classifier(texts, candidate_labels)

            # Convert to your expected format
            batch_results = []
            for result in results:
                batch_results.append({
                    "label": result["labels"][0],
                    "confidence": result["scores"][0],
                    "all_scores": dict(zip(result["labels"], result["scores"]))
                })

            return batch_results

        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            # Return unknown results for all texts
            return [{
                "label": "unknown",
                "confidence": 0.0,
                "all_scores": {label: 0.0 for label in candidate_labels}
            } for _ in texts]