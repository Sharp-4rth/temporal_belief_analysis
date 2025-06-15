"""
Simple data loaders for Reddit political data.
File: src/temporal_belief/data/loaders.py
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random


class RedditDataLoader:
    """Simple loader for Reddit political data."""

    def __init__(self, data_dir: Path, logger: logging.Logger = None):
        self.data_dir = Path(data_dir)
        self.logger = logger or logging.getLogger(__name__)

        # Create directories
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Data directories: {self.raw_dir}, {self.processed_dir}")

    def load_reddit_data(self, filename: str = "reddit_political_posts.csv") -> pd.DataFrame:
        """
        Load Reddit data from CSV file.

        Expected columns:
        - user_id: Reddit username
        - timestamp: When posted
        - text: Post content
        - subreddit: Which subreddit
        - post_id: Unique post ID (optional)

        Args:
            filename: CSV filename in data/raw/

        Returns:
            DataFrame with Reddit posts
        """
        filepath = self.raw_dir / filename

        if not filepath.exists():
            self.logger.warning(f"File not found: {filepath}")
            self.logger.info("Creating sample data for testing...")
            sample_df = self.create_sample_data()
            self.save_data(sample_df, filename)
            return sample_df

        # Load real data
        df = pd.read_csv(filepath)

        # Check required columns
        required_cols = ['user_id', 'timestamp', 'text', 'subreddit']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Clean data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.dropna(subset=['user_id', 'text'])
        df = df[df['text'].str.len() > 10]  # Remove very short posts

        # Sort by user and time
        df = df.sort_values(['user_id', 'timestamp'])

        self.logger.info(f"Loaded {len(df)} posts from {df['user_id'].nunique()} users")
        self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def create_sample_data(self) -> pd.DataFrame:
        """Create sample Reddit data for testing."""

        # Sample political texts
        liberal_texts = [
            "Healthcare should be accessible to everyone",
            "Climate change needs immediate government action",
            "We need stronger social safety nets",
            "Gun control measures are necessary",
            "Universal healthcare like other countries"
        ]

        conservative_texts = [
            "Lower taxes stimulate economic growth",
            "Free market solves problems better than government",
            "We need stronger border security",
            "Second Amendment rights must be protected",
            "Smaller government and personal responsibility"
        ]

        neutral_texts = [
            "Both parties have valid points here",
            "This issue is more complex than it seems",
            "We need bipartisan solutions",
            "Local communities should decide",
            "More research needed before policy decisions"
        ]

        # Generate sample data
        data = []
        users = [f"user_{i:03d}" for i in range(1, 21)]  # 20 users
        subreddits = ['politics', 'Conservative', 'democrats', 'Republican']

        post_id = 1
        start_date = datetime(2024, 1, 1)

        for user in users:
            # Each user posts 8-12 times over 6 months
            num_posts = random.randint(8, 12)

            for i in range(num_posts):
                # Random timestamp
                days_offset = random.randint(0, 180)
                timestamp = start_date + timedelta(
                    days=days_offset,
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )

                # Choose text based on user tendency
                user_num = int(user.split('_')[1])
                if user_num <= 8:  # Liberal users
                    texts = liberal_texts + neutral_texts
                elif user_num <= 16:  # Conservative users
                    texts = conservative_texts + neutral_texts
                else:  # Neutral users
                    texts = neutral_texts + liberal_texts + conservative_texts

                text = random.choice(texts)

                data.append({
                    'user_id': user,
                    'timestamp': timestamp,
                    'text': text,
                    'subreddit': random.choice(subreddits),
                    'post_id': f'post_{post_id:06d}',
                    'score': random.randint(0, 50)
                })
                post_id += 1

        df = pd.DataFrame(data)
        self.logger.info(f"Created sample data: {len(df)} posts from {df['user_id'].nunique()} users")

        return df

    def get_user_timelines(self, df: pd.DataFrame, min_posts: int = 5) -> Dict[str, List[Dict]]:
        """
        Organize data into user timelines.

        Args:
            df: DataFrame with posts
            min_posts: Minimum posts required per user

        Returns:
            Dict mapping user_id to list of their posts (chronological)
        """
        timelines = {}

        for user_id, user_posts in df.groupby('user_id'):
            if len(user_posts) >= min_posts:
                # Sort chronologically
                user_posts = user_posts.sort_values('timestamp')

                # Convert to list of dicts
                timeline = []
                for _, post in user_posts.iterrows():
                    timeline.append({
                        'timestamp': post['timestamp'],
                        'text': post['text'],
                        'subreddit': post['subreddit'],
                        'post_id': post.get('post_id', ''),
                        'score': post.get('score', 0)
                    })

                timelines[user_id] = timeline

        self.logger.info(f"Created {len(timelines)} user timelines (min {min_posts} posts)")
        return timelines

    def save_data(self, df: pd.DataFrame, filename: str, processed: bool = False):
        """Save DataFrame to CSV."""
        if processed:
            filepath = self.processed_dir / filename
        else:
            filepath = self.raw_dir / filename

        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved {len(df)} rows to {filepath}")

    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data with stance predictions."""
        filepath = self.processed_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Processed file not found: {filepath}")

        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        self.logger.info(f"Loaded processed data: {len(df)} rows")
        return df


def quick_test():
    """Test the data loader."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test with sample data
    data_dir = Path("data")
    loader = RedditDataLoader(data_dir, logger)

    # Load data (will create sample if doesn't exist)
    df = loader.load_reddit_data()

    print("Sample data:")
    print(df[['user_id', 'timestamp', 'text', 'subreddit']].head())

    # Test user timelines
    timelines = loader.get_user_timelines(df, min_posts=3)

    print(f"\nCreated timelines for {len(timelines)} users")

    # Show sample timeline
    if timelines:
        sample_user = list(timelines.keys())[0]
        print(f"\nSample timeline for {sample_user}:")
        for i, post in enumerate(timelines[sample_user][:3]):
            print(f"  {i + 1}. {post['timestamp']} - {post['text'][:50]}...")

    return loader, df, timelines


if __name__ == "__main__":
    loader, df, timelines = quick_test()