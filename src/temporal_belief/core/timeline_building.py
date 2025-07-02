"""Timeline building functionality for longitudinal belief analysis."""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StancePoint:
    """Single stance measurement point in a user's timeline."""
    timestamp: datetime
    stance: str
    confidence: float
    topic: str
    utterance_id: str
    conversation_id: str
    text_preview: str  # First 200 chars for debugging

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'timestamp': self.timestamp,
            'stance': self.stance,
            'confidence': self.confidence,
            'topic': self.topic,
            'utterance_id': self.utterance_id,
            'conversation_id': self.conversation_id,
            'text_preview': self.text_preview
        }


class UserTimelineBuilder:
    """Build chronological belief timelines for users from ConvoKit corpus."""

    def __init__(self, corpus, min_confidence: float = 0.5,
                 min_posts_per_topic: int = 3):
        """Initialize timeline builder.

        Args:
            corpus: ConvoKit corpus with stance and topic detection completed
            min_confidence: Minimum stance confidence to include
            min_posts_per_topic: Minimum posts needed for topic timeline
        """
        self.corpus = corpus
        self.min_confidence = min_confidence
        self.min_posts_per_topic = min_posts_per_topic
        self._user_cache = {}
        self._topic_cache = {}

        logger.info(f"Initialized timeline builder with {len(list(corpus.iter_users()))} users")

    def get_active_users(self, min_utterances: int = 10,
                         timespan_days: Optional[int] = None) -> List[str]:
        """Get users with sufficient activity for longitudinal analysis.

        Args:
            min_utterances: Minimum utterances needed
            timespan_days: If specified, only count utterances within this timespan

        Returns:
            List of user IDs suitable for longitudinal analysis
        """
        active_users = []

        for user in self.corpus.iter_users():
            user_utterances = list(user.iter_utterances())

            # Filter by timespan if specified
            if timespan_days:
                cutoff_date = datetime.now() - timedelta(days=timespan_days)
                user_utterances = [
                    utt for utt in user_utterances
                    if utt.timestamp >= cutoff_date
                ]

            # Filter by stance detection quality
            reliable_utterances = [
                utt for utt in user_utterances
                if (utt.meta.get('stance_confidence', 0) >= self.min_confidence and
                    utt.meta.get('detected_stance') != 'unknown')
            ]

            if len(reliable_utterances) >= min_utterances:
                active_users.append(user.id)

        logger.info(f"Found {len(active_users)} active users with {min_utterances}+ reliable utterances")
        return active_users

    def build_user_timeline(self, user_id: str,
                            topics: Optional[List[str]] = None) -> Dict[str, List[StancePoint]]:
        """Build chronological stance timeline for user across topics.

        Args:
            user_id: User identifier
            topics: Specific topics to analyze (if None, analyzes all user topics)

        Returns:
            Dictionary mapping topic -> chronological list of stance points
        """
        if user_id in self._user_cache:
            cached_timeline = self._user_cache[user_id]
            if topics is None:
                return cached_timeline
            else:
                # Return subset of cached timeline
                return {topic: cached_timeline.get(topic, []) for topic in topics}

        try:
            user = self.corpus.get_user(user_id)
        except:
            logger.error(f"User {user_id} not found in corpus")
            return {}

        # Get user's utterances sorted by timestamp
        user_utterances = sorted(
            list(user.iter_utterances()),
            key=lambda utt: utt.timestamp
        )

        # Determine topics to analyze
        if topics is None:
            topics = self.get_user_topics(user_id)

        timelines = defaultdict(list)

        for utterance in user_utterances:
            # Skip if stance detection failed
            if (utterance.meta.get('stance_confidence', 0) < self.min_confidence or
                    utterance.meta.get('detected_stance') == 'unknown'):
                continue

            # Get conversation topic
            try:
                conversation = self.corpus.get_conversation(utterance.conversation_id)
                topic = conversation.meta.get('detected_topic', 'unknown')
            except:
                continue

            # Skip if not in requested topics
            if topic not in topics or topic == 'unknown':
                continue

            # Create stance point
            stance_point = StancePoint(
                timestamp=utterance.timestamp,
                stance=utterance.meta['detected_stance'],
                confidence=utterance.meta['stance_confidence'],
                topic=topic,
                utterance_id=utterance.id,
                conversation_id=utterance.conversation_id,
                text_preview=utterance.text[:200] if utterance.text else ""
            )

            timelines[topic].append(stance_point)

        # Filter topics with insufficient data
        filtered_timelines = {}
        for topic, timeline in timelines.items():
            if len(timeline) >= self.min_posts_per_topic:
                filtered_timelines[topic] = timeline
            else:
                logger.debug(f"User {user_id} has only {len(timeline)} posts in {topic}, skipping")

        # Cache results
        self._user_cache[user_id] = filtered_timelines

        logger.info(f"Built timeline for user {user_id}: {len(filtered_timelines)} topics, "
                    f"{sum(len(tl) for tl in filtered_timelines.values())} total stance points")

        return filtered_timelines

    def build_multiple_timelines(self, user_ids: List[str],
                                 topics: Optional[List[str]] = None) -> Dict[str, Dict[str, List[StancePoint]]]:
        """Build timelines for multiple users efficiently.

        Args:
            user_ids: List of user identifiers
            topics: Specific topics to analyze

        Returns:
            Dictionary mapping user_id -> topic -> timeline
        """
        all_timelines = {}

        for user_id in user_ids:
            try:
                timeline = self.build_user_timeline(user_id, topics)
                if timeline:  # Only include users with valid timelines
                    all_timelines[user_id] = timeline
            except Exception as e:
                logger.error(f"Failed to build timeline for user {user_id}: {e}")
                continue

        logger.info(f"Built timelines for {len(all_timelines)} users")
        return all_timelines

    def timeline_to_dataframe(self, timeline: List[StancePoint]) -> pd.DataFrame:
        """Convert timeline to pandas DataFrame for analysis.

        Args:
            timeline: List of stance points

        Returns:
            DataFrame with columns: timestamp, stance, confidence, etc.
        """
        if not timeline:
            return pd.DataFrame()

        data = [point.to_dict() for point in timeline]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')

    def get_timeline_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary statistics for user's timeline.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with timeline summary statistics
        """
        timelines = self.build_user_timeline(user_id)

        if not timelines:
            return {'user_id': user_id, 'active_topics': 0, 'total_posts': 0}

        total_posts = sum(len(timeline) for timeline in timelines.values())

        # Calculate timespan
        all_timestamps = []
        for timeline in timelines.values():
            all_timestamps.extend([point.timestamp for point in timeline])

        if all_timestamps:
            timespan_days = (max(all_timestamps) - min(all_timestamps)).days
            avg_confidence = sum(
                point.confidence
                for timeline in timelines.values()
                for point in timeline
            ) / total_posts
        else:
            timespan_days = 0
            avg_confidence = 0.0

        return {
            'user_id': user_id,
            'active_topics': len(timelines),
            'total_posts': total_posts,
            'timespan_days': timespan_days,
            'avg_confidence': avg_confidence,
            'topics': list(timelines.keys())
        }

    def export_timelines(self, user_ids: List[str],
                         output_path: str, format: str = 'csv') -> None:
        """Export timelines to file for external analysis.

        Args:
            user_ids: List of user identifiers
            output_path: Path to save file
            format: Export format ('csv', 'json', 'parquet')
        """
        all_data = []

        for user_id in user_ids:
            timelines = self.build_user_timeline(user_id)

            for topic, timeline in timelines.items():
                for point in timeline:
                    row = point.to_dict()
                    row['user_id'] = user_id
                    all_data.append(row)

        df = pd.DataFrame(all_data)

        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', date_format='iso')
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(all_data)} stance points to {output_path}")