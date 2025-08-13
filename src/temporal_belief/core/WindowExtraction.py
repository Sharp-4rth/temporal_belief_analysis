

class WindowExtractor:
    """ Find the conversations around the change point """
    def __init__(self, corpus, timelines):
        self.corpus = corpus
        self.timelines = timelines

    def get_user_conversations_chronological(self, corpus, speaker_id):
        """Get all conversations for a user in chronological order."""

        # Get all conversations where the speaker participated
        user_conversations = [convo for convo in corpus.iter_conversations()
                              if speaker_id in [utt.speaker.id for utt in convo.iter_utterances()]]

        # Sort conversations by their earliest timestamp
        user_conversations.sort(key=lambda convo: min(utt.timestamp for utt in convo.iter_utterances()))

        return user_conversations

    def get_conversations_around_change_point(self, corpus, change_point):
        # Get first change (probably only one I need)
        utterance = corpus.get_utterance(change_point[0][1])

        # Find the convo this utterance belongs to:
        conversation = utterance.get_conversation()

        # Put all user's convos in a list
        speaker_id = utterance.speaker.id
        user_conversations = self.get_user_conversations_chronological(corpus, speaker_id)

        candidate_convos = []
        # find the index of the convo, and return the convo id of the 3 prior convos
        for i, convo in enumerate(user_conversations):
            if conversation.id == user_conversations[i].id:
                candidate_convos.append(user_conversations[i - 2])
                candidate_convos.append(user_conversations[i - 1])

        # Append the first convo at the end so they are in chronological order
        candidate_convos.append(conversation)

        return candidate_convos