

# Mock corpus-like objects for testing
class MockUtterance:
    def __init__(self, id, speaker_id, text, timestamp, conversation):
        self.id = id
        self.speaker = MockSpeaker(speaker_id)
        self.text = text
        self.timestamp = timestamp
        self._conversation = conversation

    def get_conversation(self):
        return self._conversation

class MockSpeaker:
    def __init__(self, speaker_id):
        self.id = speaker_id

class MockConversation:
    def __init__(self, id, root_post, all_utterances):
        self.id = id
        self._root = root_post
        self._utterances = all_utterances
        self.meta = {'detected_topic': 'taxation and government spending'}

    def get_root(self):
        return self._root

    def iter_utterances(self):
        return iter(self._utterances)

class MockCorpus:
    def __init__(self):
        # Create mock conversations with realistic political discussion
        self.utterances = {}
        self.conversations = {}
        self._setup_mock_data()

    def get_utterance(self, utterance_id):
        return self.utterances.get(utterance_id)

    def _setup_mock_data(self):
        # Create 3 conversations for testing

        # Conversation 1: Tax policy discussion
        conv1_utterances = []
        op1 = MockUtterance(
            id="op_conv1",
            speaker_id="original_poster_1",
            text="I think we should raise taxes on the wealthy to fund infrastructure. The current system isn't working.",
            timestamp=None,
            conversation=None
        )

        user_reply1 = MockUtterance(
            id="user_reply_conv1",
            speaker_id="TestUser",
            text="I disagree with raising taxes. The wealthy already pay their fair share and higher taxes will hurt economic growth.",
            timestamp=None,
            conversation=None
        )

        conv1_utterances = [op1, user_reply1]
        conv1 = MockConversation("conv_1", op1, conv1_utterances)

        # Set conversation references
        for utt in conv1_utterances:
            utt._conversation = conv1

        # Conversation 2: Healthcare spending
        op2 = MockUtterance(
            id="op_conv2",
            speaker_id="original_poster_2",
            text="Government healthcare spending is out of control. We need to cut Medicare and focus on private solutions.",
            timestamp=None,
            conversation=None
        )

        user_reply2 = MockUtterance(
            id="user_reply_conv2",
            speaker_id="TestUser",
            text="Medicare is essential for seniors. We shouldn't cut it, but maybe we can find efficiencies without reducing benefits.",
            timestamp=None,
            conversation=None
        )

        conv2_utterances = [op2, user_reply2]
        conv2 = MockConversation("conv_2", op2, conv2_utterances)

        for utt in conv2_utterances:
            utt._conversation = conv2

        # Conversation 3: Budget discussion
        op3 = MockUtterance(
            id="op_conv3",
            speaker_id="original_poster_3",
            text="The federal budget deficit is unsustainable. We need major spending cuts across all departments.",
            timestamp=None,
            conversation=None
        )

        user_reply3 = MockUtterance(
            id="user_reply_conv3",
            speaker_id="TestUser",
            text="You're absolutely right. Government spending is completely out of control and we need dramatic cuts immediately.",
            timestamp=None,
            conversation=None
        )

        conv3_utterances = [op3, user_reply3]
        conv3 = MockConversation("conv_3", op3, conv3_utterances)

        for utt in conv3_utterances:
            utt._conversation = conv3

        # Store all utterances and conversations
        all_utterances = conv1_utterances + conv2_utterances + conv3_utterances
        for utt in all_utterances:
            self.utterances[utt.id] = utt

        self.conversations["conv_1"] = conv1
        self.conversations["conv_2"] = conv2
        self.conversations["conv_3"] = conv3

# Mock timeline data showing a clear stance change
mock_timelines = {
    "TestUser": {
        "taxation and government spending": {
            "user_reply_conv1": "moderately_against",  # Position 0: Against tax increases
            "user_reply_conv2": "neutral",             # Position 1: Moderate on spending
            "user_reply_conv3": "strongly_against"     # Position 2: Strong anti-spending (CHANGE HERE)
        }
    }
}

# Mock significant change (what your detector would return)
mock_significant_change = {
    'position': 2,
    'utterance_id': 'user_reply_conv3',  # The utterance where change was detected
    'p_value': 0.023,
    'p_corrected': 0.041,
    'magnitude': 1.5,
    'left_mean': -0.5,   # Was moderate
    'right_mean': -2.0,  # Became strongly against
    'statistically_significant': True,
    'survives_fdr_correction': True
}

print("Mock data created!")
print("Timeline:", mock_timelines["TestUser"]["taxation and government spending"])
print("Significant change detected at:", mock_significant_change['utterance_id'])