from temporal_belief.core.window_extraction import WindowExtractor



class OpPathPairer:
    """ Pair OP utterances with a path of responses by a user/challenger"""
    def __init__(self, corpus, timelines):
        self.corpus = corpus
        self.timelines = timelines

    def _trim_paths(self, op_utterance):
        try:
            conversation = op_utterance.get_conversation()
        except Exception as e:
            print(f"Can't access convo from utterance, error{e}")

        paths = conversation.get_root_to_leaf_paths()

        trimmed_paths = []
        for path in paths:
            if op_utterance in path:
                # Find where op_utterance is in this path
                op_index = path.index(op_utterance)
                # Slice from that index onwards
                trimmed_path = path[op_index + 1:]
                trimmed_paths.append(trimmed_path)

        return trimmed_paths

    def _filter_paths(self, trimmed_paths, op_speaker_id):
        """Filter paths to create rooted path-units, excluding OP utterances"""
        filtered_paths = {}

        for path_index, path in enumerate(trimmed_paths):
            for utt in path:
                # Skip if this utterance is from the OP
                if utt.speaker.id == op_speaker_id:
                    continue

                key = f"{utt.speaker.id}_path_{path_index}"
                if key not in filtered_paths:
                    filtered_paths[key] = []
                filtered_paths[key].append(utt)

        return filtered_paths

    def extract_rooted_paths(self, op_utterance):
        trimmed_path = self._trim_paths(op_utterance)
        # Pass the OP's speaker ID to filter method
        filtered_path = self._filter_paths(trimmed_path, op_utterance.speaker.id)

        return filtered_path

    # Find the op_utterances from a convo and add them to a list
    def extract_op_utterances_from_convo(self, candidate_convo, user_id):
        paths = candidate_convo.get_root_to_leaf_paths()
        op_utterances = []
        for path in paths:
            for utt in path:
                if utt.speaker.id == user_id and utt not in op_utterances:
                    op_utterances.append(utt)
                    break

        return op_utterances

    # Get all op_utterances accross every candidate convo
    def extract_op_utterances_from_all_convos(self, candidate_convos, user_id):
        all_op_utterances = []
        for candidate_convo in candidate_convos:
            op_utterances = self.extract_op_utterances_from_convo(candidate_convo, user_id)
            all_op_utterances.extend(op_utterances)

        return all_op_utterances

    # Get the paths of an op_utterance from the op_utterances list
    def extract_rooted_path_from_candidate_convos(self, candidate_convos, user_id):
        all_op_utterances = self.extract_op_utterances_from_all_convos(candidate_convos, user_id)

        # debug:
        # for op_utt in all_op_utterances:
        #     print(f'my input user_id: {user_id}')
        #     speaker_id = self.corpus.get_utterance(op_utt.id).speaker.id
        #     print(f'Utt_id: {op_utt.id} and user_id: {speaker_id} in the list of all op utterances.')

        all_ops_n_paths = []
        for op_utt in all_op_utterances:
            # So rooted paths is a dict. Should I convert to list?
            rooted_paths = self.extract_rooted_paths(op_utt)

            op_n_paths = (op_utt, rooted_paths)
            all_ops_n_paths.append(op_n_paths)

        return all_ops_n_paths