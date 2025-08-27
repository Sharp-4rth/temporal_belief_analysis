from typing import Dict, Any
from collections import defaultdict
import logging
from ..utils.config import MERGED_TOPIC

class TimelineBuilder:
    """Simple timeline builder for user belief tracking.

    Builds structure: {user_id: {topic: {utterance_id: stance}}}
    """

    def __init__(self, corpus, min_posts_per_topic: int = 0, min_topics_per_user: int = 0):
        self.corpus = corpus
        self.min_posts_per_topic = min_posts_per_topic
        self.min_topics_per_user = min_topics_per_user
        self.logger = logging.getLogger(__name__)

    def build_timelines(self, include_all=True) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Build user timelines from corpus with stance metadata.

        Returns:
            {user_id: {topic: {utterance_id: stance}}}
        """
        # Group by user and topic
        user_topic_posts = defaultdict(lambda: defaultdict(list))

        for utterance in self.corpus.iter_utterances():
            # Skip if no stance metadata on utterance
            if include_all == False:
                if not utterance.meta or 'detected_stance' not in utterance.meta:
                    continue

            # Get topic from conversation metadata
            conversation = utterance.get_conversation()
            if not conversation or not conversation.meta or 'detected_topic' not in conversation.meta:
                continue

            if not utterance.timestamp:
                continue

            user_id = utterance.speaker.id
            old_topic = conversation.meta['detected_topic']
            topic = MERGED_TOPIC.get(old_topic, old_topic)
            stance = utterance.meta.get('detected_stance', 'Unknown')

            user_topic_posts[user_id][topic].append({
                'utterance_id': utterance.id,
                'timestamp': utterance.timestamp,
                'stance': stance
            })

        # Filter and sort
        timelines = {}
        for user_id, topic_posts in user_topic_posts.items():
            user_timeline = {}

            for topic, posts in topic_posts.items():
                if len(posts) >= self.min_posts_per_topic:
                    # Sort chronologically
                    posts.sort(key=lambda x: x['timestamp'])

                    # Create topic timeline
                    topic_timeline = {}
                    for post in posts:
                        topic_timeline[post['utterance_id']] = post['stance']

                    user_timeline[topic] = topic_timeline

            # Only include users with enough topics
            if len(user_timeline) >= self.min_topics_per_user:
                timelines[user_id] = user_timeline

        self.logger.info(f"Built timelines for {len(timelines)} users")
        return timelines