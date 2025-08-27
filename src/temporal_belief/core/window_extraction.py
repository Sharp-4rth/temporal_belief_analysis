from tqdm import tqdm

class WindowExtractor:
    """ Find the conversations around the change point """

    def __init__(self, corpus, timelines):
        self.corpus = corpus
        self.timelines = timelines
        self.user_conversations_cache = {}  # Add cache

    def build_global_user_conversations_index(self):
        """Build sorted conversations for ALL users upfront"""
        print("Building global user conversations index...")
        user_conversations = {}

        convos = list(corpus.iter_conversations())
        print(f"Processing {len(convos)} conversations...")

        for convo in convos:
            # Get all speakers in this conversation
            speakers = {utt.speaker.id for utt in convo.iter_utterances()}

            # Add this conversation to each speaker's list
            for speaker_id in speakers:
                if speaker_id not in user_conversations:
                    user_conversations[speaker_id] = []
                user_conversations[speaker_id].append(convo)

        # Sort each user's conversations once
        print(f"Sorting conversations for {len(user_conversations)} users...")
        for speaker_id in user_conversations:
            user_conversations[speaker_id].sort(
                key=lambda convo: min(utt.timestamp for utt in convo.iter_utterances())
            )

        print(f"Index built for {len(user_conversations)} users!")

        self.user_conversations_cache = user_conversations

    def get_user_conversations_chronological_old(self, corpus, speaker_id):
        """Get all conversations for a user in chronological order."""

        # Check cache first
        if speaker_id in self.user_conversations_cache:
            return self.user_conversations_cache[speaker_id]

        # Get all conversations where the speaker participated
        user_conversations = [convo for convo in corpus.iter_conversations()
                              if speaker_id in [utt.speaker.id for utt in convo.iter_utterances()]]

        # Sort conversations by their earliest timestamp
        user_conversations.sort(key=lambda convo: min(utt.timestamp for utt in convo.iter_utterances()))

        # Cache the result
        self.user_conversations_cache[speaker_id] = user_conversations

        return user_conversations

    def get_user_conversations_chronological(self, corpus, speaker_id):
        return self.user_conversations_cache.get(speaker_id, [])

    def get_conversations_around_change_point(self, corpus, change_point, test=False, window=10):
        # Get first change (probably only one I need)
        utterance = corpus.get_utterance(change_point)

        # Find the convo this utterance belongs to:
        conversation = utterance.get_conversation()

        # Put all user's convos in a list
        speaker_id = utterance.speaker.id
        if test is True:
            user_conversations = self.get_user_conversations_chronological_old(corpus, speaker_id)
        else:
            user_conversations = self.get_user_conversations_chronological(corpus, speaker_id)
            # print(f"Cache: {user_conversations}")

        candidate_convos = []
        # find the index of the convo, and return the convo id of the 3 prior convos
        for i, convo in enumerate(user_conversations):
            if conversation.id == user_conversations[i].id:
                # Check if there are at least two conversations before the current one
                # To this:
                if i >= window:
                    # Get the 'window' number of conversations before the current one
                    candidate_convos.extend(user_conversations[i - 10:i])
                else:
                    # If there are fewer than 10 conversations before, get all of them
                    candidate_convos.extend(user_conversations[:i])

                # Append the current conversation with the change point
                candidate_convos.append(conversation)
                break  # Found the conversation, no need to continue the loop

        return candidate_convos