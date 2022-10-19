from nltk import word_tokenize
from nltk.corpus import stopwords

from src.utils import get_targets

characters = ["",'']


def get_queries_and_targets_from_candidates(candidate_dictionary, corpus_path, target_fields):
    targets = get_targets(corpus_path, target_fields) #added this
    candidates_per_query = {}
    query_ids = list(candidate_dictionary.keys())
    for query_id in query_ids:
        candidates_ids_and_texts_per_query = {}
        candidate_targets = candidate_dictionary[query_id]
        candidate_targets_ids = list(candidate_targets.keys())
        for candidate_target_id in candidate_targets_ids:
            candidates_ids_and_texts_per_query[candidate_target_id] = targets[candidate_target_id]
        candidates_per_query[query_id] = candidates_ids_and_texts_per_query
    return candidates_per_query


def get_all_relevant_targets(candidates_per_query):
    candidates_dict = {}
    for candidates in list(candidates_per_query.values()):
        candidates_dict = candidates_dict | candidates
    return candidates_dict


def tokenize_and_filter_out_stop_words(sequence):
    stop_words = set(stopwords.words('english'))
    return [w for w in tokenize(sequence) if not w.lower() in stop_words and not w in characters and len(w) > 1]


def tokenize(sequence):
    return word_tokenize(sequence)







