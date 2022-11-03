
# def get_sequence_similarity(queries, candidate_queries_and_targets):
#     sequence_similarities = {}
#     for query_id, target_dict in candidate_queries_and_targets.items():
#         query_text = queries[query_id]
#         query_words = set(tokenize_and_filter_out_stop_words(query_text))
#         query_word_number = len(query_words)
#         target_sims = {}
#         for target_id, target_text in target_dict.items():
#             target_words = set(tokenize_and_filter_out_stop_words(target_text))
#             target_words_number = len(target_words)
#             pair_length = query_word_number + target_words_number
#             common_words = set.intersection(set(query_words), set(target_words))
#             common_words_number = len(common_words)
#             if common_words_number > 0:
#                 target_sims[target_id] = (1/pair_length)*2*common_words_number
#             else:
#                 target_sims[target_id] = 0
#         sequence_similarities[query_id] = target_sims
#     return sequence_similarities
#
#     sequence_comp = SequenceMatcher(a=string_1[0], b=string_2)
#     return sequence_comp.ratio()