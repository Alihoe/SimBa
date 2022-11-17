from difflib import SequenceMatcher
from nltk import word_tokenize
from Levenshtein import ratio


def match_sequences(query_text, target_text):
    return (SequenceMatcher(a=query_text, b=target_text).ratio())*100


def get_sequence_matching_similarity(queries, candidate_queries_and_targets):
    sequence_similarities = {}
    for query_id, target_dict in candidate_queries_and_targets.items():
        query_text = queries[query_id]
        target_sims = {}
        for target_id, target_text in target_dict.items():
            target_sims[target_id] = match_sequences(query_text, target_text)
        sequence_similarities[query_id] = target_sims
    print(list(sequence_similarities.items())[0])
    return sequence_similarities


def levenshtein_sim(query_text, target_text):
    return ratio(query_text, target_text)*100


def get_levenshtein_similarity(queries, candidate_queries_and_targets):
    levenshtein_similarities = {}
    for query_id, target_dict in candidate_queries_and_targets.items():
        query_text = queries[query_id]
        target_sims = {}
        for target_id, target_text in target_dict.items():
            target_sims[target_id] = levenshtein_sim(query_text, target_text)
        levenshtein_similarities[query_id] = target_sims
    print(list(levenshtein_similarities.items())[0])
    return levenshtein_similarities


def jac_sim(query_text, target_text):
    a = set(word_tokenize(query_text))
    b = set(word_tokenize(target_text))
    return (float(len(a.intersection(b))) / len(a.union(b)))*100


def get_jacquard_similarity(queries, candidate_queries_and_targets):
    jacquard_similarities = {}
    for query_id, target_dict in candidate_queries_and_targets.items():
        query_text = queries[query_id]
        target_sims = {}
        for target_id, target_text in target_dict.items():
            target_sims[target_id] = jac_sim(query_text, target_text)
        jacquard_similarities[query_id] = target_sims
    print(list(jacquard_similarities.items())[0])
    return jacquard_similarities


