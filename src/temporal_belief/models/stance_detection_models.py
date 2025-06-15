"""
Simple stance detection using CardiffNLP (SemEval) pre-trained model.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict, List, Any


class StanceDetector:
    """Simple stance detection using pre-trained SemEval model."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        # CardiffNLP
        model_name = "cardiffnlp/twitter-roberta-base-stance-detection"

        self.logger.info(f"Loading stance detection model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Model outputs: 0=against, 1=neutral, 2=favor
        self.labels = {0: "against", 1: "neutral", 2: "favor"}

        self.logger.info("Stance detection model loaded successfully")

    def predict_stance(self, text: str) -> Dict[str, Any]:
        """
        Predict stance for a single text.

        Args:
            text: Input text to analyze

        Returns:
            Dict with stance, confidence, and other info
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Extract results
        confidence = torch.max(probs).item()
        prediction_id = torch.argmax(probs).item()
        stance = self.labels[prediction_id]

        return {
            'text': text,
            'stance': stance,
            'stance_id': prediction_id,
            'confidence': confidence,
            'reliable': confidence > 0.7,  # Confidence threshold
            'probabilities': probs.cpu().numpy().tolist()[0]
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict stance for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of prediction dictionaries
        """
        results = []

        self.logger.info(f"Processing {len(texts)} texts...")

        for i, text in enumerate(texts):
            result = self.predict_stance(text)
            results.append(result)

            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(texts)} texts")

        self.logger.info(f"Completed batch prediction for {len(texts)} texts")
        return results


def quick_test():
    """Quick test function to verify the model works."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    detector = StanceDetector(logger)

    test_texts = [
        "Healthcare should be free for everyone",
        "Lower taxes help the economy grow",
        "Climate change needs immediate action",
        "The free market solves problems best",
        "We need stronger social programs"
    ]

    print("Testing Stance Detection:")
    print("=" * 50)

    for text in test_texts:
        result = detector.predict_stance(text)
        print(f"Text: {text}")
        print(f"Stance: {result['stance']} (confidence: {result['confidence']:.3f})")
        print(f"Reliable: {result['reliable']}")
        print("-" * 30)

    return detector


if __name__ == "__main__":
    detector = quick_test()