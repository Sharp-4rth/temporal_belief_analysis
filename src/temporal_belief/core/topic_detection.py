"""Topic detection functionality for conversation analysis."""

from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm

from ..models.bart_classifier import BARTZeroShotClassifier
from ..utils.config import POLITICAL_TOPICS

logger = logging.getLogger(__name__)

class TopicDetector:
    """Detect topics in ConvoKit conversations using BART."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli",
                 topics: Optional[List[str]] = None):
        """Initialize topic detector."""
        self.classifier = BARTZeroShotClassifier(model_name)
        self.topics = topics or POLITICAL_TOPICS
        logger.info(f"Initialized topic detector with {len(self.topics)} topics")

    def detect_conversation_topic(self, conversation) -> Dict[str, Any]:
        """Detect topic for a single conversation."""
        utterances = list(conversation.iter_utterances())
        if not utterances:
            logger.warning(f"No utterances found in conversation {conversation.id}")
            return {"topic": "unknown", "confidence": 0.0}

        original_post = utterances[0].text
        result = self.classifier.classify_text(original_post, self.topics)

        return {
            "topic": result["label"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
            "text_length": len(original_post),
            "num_utterances": len(utterances)
        }

    def process_corpus(self, corpus, batch_size: int = 50,
                       save_path: Optional[str] = None) -> None:
        """Process entire corpus for topic detection."""
        conversations = list(corpus.iter_conversations())
        logger.info(f"Processing {len(conversations)} conversations for topic detection")

        for i in tqdm(range(0, len(conversations), batch_size),
                      desc="Processing conversations"):
            batch = conversations[i:i + batch_size]

            for conv in batch:
                try:
                    topic_result = self.detect_conversation_topic(conv)

                    # Add to conversation metadata
                    conv.add_meta("detected_topic", topic_result["topic"])
                    conv.add_meta("topic_confidence", topic_result["confidence"])
                    conv.add_meta("topic_scores", topic_result["all_scores"])

                except Exception as e:
                    logger.error(f"Failed to process conversation {conv.id}: {e}")
                    conv.add_meta("detected_topic", "unknown")
                    conv.add_meta("topic_confidence", 0.0)

        if save_path:
            corpus.dump(save_path)
            logger.info(f"Saved processed corpus to {save_path}")

        logger.info("Topic detection processing complete")