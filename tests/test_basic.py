# User to test:
user_id = "HardCoreModerate"
topic = "economic policy"

topic_timeline = timelines[user_id][topic]
topic_timeline_list = list(topic_timeline.items())
change_points = change_detector.detect_cusum_changes(topic_timeline_list)['change_points']
print(change_points)

candidate_convos = window_extractor.get_conversations_around_change_point(change_point=change_points[0], corpus=corpus, test=True)
print({[convo for convo in candidate_convos]}")

op_path_pairs = op_path_pairer.extract_rooted_path_from_candidate_convos(candidate_convos, user_id)
print(op_path_pairs)

preprocessed_pairs = pair_preprocessor.concatenate_path_in_all_pairs(op_path_pairs)
print(preprocessed_pairs)"""
Basic function tests
"""