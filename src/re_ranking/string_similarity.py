from difflib import SequenceMatcher
from nltk import word_tokenize
from Levenshtein import ratio


def get_sequence_matching_similarity(queries, candidate_queries_and_targets):
    sequence_similarities = {}
    for query_id, target_dict in candidate_queries_and_targets.items():
        query_text = queries[query_id]
        target_sims = {}
        for target_id, target_text in target_dict.items():
            target_sims[target_id] = SequenceMatcher(a=query_text, b=target_text)*100
        sequence_similarities[query_id] = target_sims
    print(sequence_similarities)
    return sequence_similarities


def get_levenshtein_similarity(queries, candidate_queries_and_targets):
    levenshtein_similarities = {}
    for query_id, target_dict in candidate_queries_and_targets.items():
        query_text = queries[query_id]
        target_sims = {}
        for target_id, target_text in target_dict.items():
            target_sims[target_id] = ratio(a=query_text, b=target_text)*100
        levenshtein_similarities[query_id] = target_sims
    print(levenshtein_similarities)
    return levenshtein_similarities


def get_jacquard_similarity(queries, candidate_queries_and_targets):
    jacquard_similarities = {}
    for query_id, target_dict in candidate_queries_and_targets.items():
        query_text = queries[query_id]
        target_sims = {}
        for target_id, target_text in target_dict.items():
            a = set(word_tokenize(query_text))
            b = set(word_tokenize(target_text))
            target_sims[target_id] = (float(len(a.intersection(b))) / len(a.union(b)))*100
        jacquard_similarities[query_id] = target_sims
    print(jacquard_similarities)
    return jacquard_similarities


