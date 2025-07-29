"""BART-based zero-shot classification for topic and stance detection."""

from transformers import pipeline
from typing import List, Dict, Any, Optional
import logging
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BARTZeroShotClassifier:
    """BART-based zero-shot classifier with multi-GPU support."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli", num_gpus: int = 1):
        """Initialize BART classifier.

        Args:
            model_name: HuggingFace model identifier
            num_gpus: Number of GPUs to use
        """
        self.model_name = model_name
        self.num_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        self.classifiers = []
        self._load_models()

    def _load_models(self):
        """Load BART models across multiple GPUs."""
        try:
            if torch.cuda.is_available() and self.num_gpus > 1:
                # Create one classifier per GPU
                for gpu_id in range(self.num_gpus):
                    classifier = pipeline(
                        task="zero-shot-classification",
                        model=self.model_name,
                        device=gpu_id
                    )
                    self.classifiers.append(classifier)
                    logger.info(f"Loaded classifier on GPU {gpu_id}")
            else:
                # Single GPU or CPU fallback
                device = 0 if torch.cuda.is_available() else -1
                classifier = pipeline(
                    task="zero-shot-classification",
                    model=self.model_name,
                    device=device
                )
                self.classifiers.append(classifier)
                logger.info(f"Loaded single classifier on device: {'GPU' if device >= 0 else 'CPU'}")

        except Exception as e:
            logger.error(f"Failed to load BART model: {e}")
            raise

    def classify_text(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Classify text into one of the candidate labels."""
        if not text.strip():
            return {
                "label": "unknown",
                "confidence": 0.0,
                "all_scores": {label: 0.0 for label in candidate_labels}
            }

        try:
            # Use first GPU for single classification
            result = self.classifiers[0](text, candidate_labels)
            return {
                "label": result["labels"][0],
                "confidence": result["scores"][0],
                "all_scores": dict(zip(result["labels"], result["scores"]))
            }
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "label": "unknown",
                "confidence": 0.0,
                "all_scores": {label: 0.0 for label in candidate_labels}
            }

    def classify_batch_parallel(self, text_hypothesis_pairs: List[tuple]) -> List[Dict[str, Any]]:
        """Classify multiple (text, hypothesis) pairs in parallel across GPUs."""
        if not text_hypothesis_pairs:
            return []

        if len(self.classifiers) == 1:
            # Single GPU fallback
            return [self._classify_single_pair(text, hypothesis, 0)
                   for text, hypothesis in text_hypothesis_pairs]

        results = [None] * len(text_hypothesis_pairs)

        def classify_task(task_data):
            idx, text, hypothesis, gpu_idx = task_data
            return idx, self._classify_single_pair(text, hypothesis, gpu_idx)

        # Assign tasks to GPUs in round-robin
        tasks = []
        for i, (text, hypothesis) in enumerate(text_hypothesis_pairs):
            gpu_idx = i % len(self.classifiers)
            tasks.append((i, text, hypothesis, gpu_idx))

        # Process in parallel
        with ThreadPoolExecutor(max_workers=len(self.classifiers)) as executor:
            future_to_task = {executor.submit(classify_task, task): task for task in tasks}

            for future in as_completed(future_to_task):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Task failed: {e}")

        # Fill any None results
        for i, result in enumerate(results):
            if result is None:
                text, hypothesis = text_hypothesis_pairs[i]
                results[i] = {
                    "label": "unknown",
                    "confidence": 0.0,
                    "all_scores": {hypothesis: 0.0}
                }

        return results

    def _classify_single_pair(self, text: str, hypothesis: str, gpu_idx: int) -> Dict[str, Any]:
        """Classify a single (text, hypothesis) pair on specific GPU."""
        try:
            classifier = self.classifiers[gpu_idx]
            result = classifier(text, [hypothesis])
            return {
                "label": result["labels"][0],
                "confidence": result["scores"][0],
                "all_scores": dict(zip(result["labels"], result["scores"]))
            }
        except Exception as e:
            logger.error(f"Classification failed on GPU {gpu_idx}: {e}")
            return {
                "label": "unknown",
                "confidence": 0.0,
                "all_scores": {hypothesis: 0.0}
            }