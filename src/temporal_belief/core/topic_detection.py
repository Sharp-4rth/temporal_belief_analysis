"""Topic detection functionality for conversation analysis."""

from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
from ..models.bart_classifier import BARTZeroShotClassifier
from ..utils.config import POLITICAL_TOPICS, ProjectConfig
from temporal_belief.utils.logger import setup_logging

logger = setup_logging("DEBUG")

class TopicDetector:
    """Detect topics in ConvoKit conversations using BART."""

    def __init__(self, topics: Optional[List[str]] = None,
                 config: ProjectConfig = None):
        """Initialize topic detector."""
        self.config = config or ProjectConfig()
        self.classifier = BARTZeroShotClassifier(self.config.bart_model_name)
        self.topics = topics or POLITICAL_TOPICS
        logger.info(f"Initialized topic detector with {len(self.topics)} topics")

    def detect_conversation_topic(self, conversation) -> Dict[str, Any]:
        """Detect topic for a single conversation."""
        utterances = list(conversation.iter_utterances())

        # Safe attribute access
        title = conversation.meta.get('title', '')

        # Safe utterance handling
        first_utterance = utterances[0] if utterances else None
        original_post = first_utterance.text if first_utterance else ''

        if not original_post and not title:
            logger.warning(f"No utterances or title found in conversation {conversation.id}")
            return {"topic": "unknown", "confidence": 0.0}

        # Truncate long texts to prevent memory issues
        combined_text = f"Title: {title}. Original Post: {original_post}"[:2000]
        result = self.classifier.classify_text(combined_text, self.topics)

        return {
            "topic": result["label"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
            "text_length": len(original_post),
            "num_utterances": len(utterances)
        }

    def process_corpus(self, corpus, batch_size: int = 50,  # Balanced batch size
                       save_path: Optional[str] = None) -> None:
        """Process entire corpus for topic detection."""
        conversations = list(corpus.iter_conversations())
        logger.info(f"Processing {len(conversations)} conversations for topic detection")

        for i in tqdm(range(0, len(conversations), batch_size),
                      desc="Processing conversations"):
            batch = conversations[i:i + batch_size]

            # Prepare all texts for batch processing
            batch_texts = []
            valid_conversations = []

            for conv in batch:
                try:
                    # Safe attribute access
                    title = conv.meta.get('title', '')
                    utterances = list(conv.iter_utterances())

                    # Safe utterance handling
                    first_utterance = utterances[0] if utterances else None
                    original_post = first_utterance.text if first_utterance else ''

                    if not original_post and not title:
                        logger.warning(f"No utterances or title found in conversation {conv.id}")
                        # Set metadata for empty conversations
                        conv.add_meta("detected_topic", "unknown")
                        conv.add_meta("topic_confidence", 0.0)
                        conv.add_meta("topic_scores", {})
                        continue

                    # Truncate long texts
                    combined_text = f"{title}. {original_post}"[:2000]
                    batch_texts.append(combined_text)
                    valid_conversations.append(conv)

                except Exception as e:
                    logger.error(f"Failed to prepare conversation {conv.id}: {e}")
                    conv.add_meta("detected_topic", "unknown")
                    conv.add_meta("topic_confidence", 0.0)
                    conv.add_meta("topic_scores", {})

            # Process entire batch at once
            if batch_texts:
                try:
                    print(f"🚀 Attempting batch of {len(batch_texts)} texts...")
                    import time
                    start = time.time()

                    batch_results = self.classifier.classify_batch(batch_texts, self.topics)

                    end = time.time()
                    print(f"✅ Batch completed in {end - start:.2f}s ({(end - start) / len(batch_texts):.3f}s per text)")

                    # Apply results back to conversations
                    for conv, result in zip(valid_conversations, batch_results):
                        conv.add_meta("detected_topic", result["label"])
                        conv.add_meta("topic_confidence", result["confidence"])
                        conv.add_meta("topic_scores", result["all_scores"])

                except Exception as e:
                    print(f"❌ Batch processing failed: {e}")
                    logger.error(f"Batch classification failed: {e}")

                    # Fallback to individual processing
                    for conv in valid_conversations:
                        try:
                            topic_result = self.detect_conversation_topic(conv)
                            conv.add_meta("detected_topic", topic_result["topic"])
                            conv.add_meta("topic_confidence", topic_result["confidence"])
                            conv.add_meta("topic_scores", topic_result["all_scores"])
                        except Exception as e2:
                            logger.error(f"Individual fallback failed for {conv.id}: {e2}")
                            conv.add_meta("detected_topic", "unknown")
                            conv.add_meta("topic_confidence", 0.0)
                            conv.add_meta("topic_scores", {})

        if save_path:
            corpus.dump(save_path)
            logger.info(f"Saved processed corpus to {save_path}")

        logger.info("Topic detection processing complete")