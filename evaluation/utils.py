from src.re_ranking.ranking_utils import tokenize_and_filter_out_stop_words
from src.re_ranking.referential_similarity import ne_sim, get_synonym_ratio
from src.re_ranking.string_similarity import match_sequences, levenshtein_sim, jac_sim
from src.sentence_encoder import encode_queries, encode_targets
from scipy.spatial.distance import cdist
import numpy as np


def get_sim_score(feature, query_text, target_text, similarity_measure):

    if feature == "similar_words_ratio":
        query_words = set(tokenize_and_filter_out_stop_words(query_text))
        query_word_number = len(query_words)
        target_words = set(tokenize_and_filter_out_stop_words(target_text))
        target_words_number = len(target_words)
        pair_length = query_word_number + target_words_number
        common_words = set.intersection(set(query_words), set(target_words))
        common_words_number = len(common_words)
        if common_words_number > 0:
            sim_score = ((1 / pair_length) * 2 * common_words_number)*100
        else:
            sim_score = 0

    elif feature == "sequence_matching_similarity":
        sim_score = match_sequences(query_text, target_text)

    elif feature == "levenshtein_similarity":
        sim_score = levenshtein_sim(query_text, target_text)

    elif feature == "jacquard_similarity":
        sim_score = jac_sim(query_text, target_text)

    elif feature == "synonym_similarity":
        sim_score = get_synonym_ratio(query_text, target_text)

    elif feature == "ne_similarity":
        sim_score = ne_sim(query_text, target_text)

    else:
        embedded_queries = encode_queries({'query': query_text}, feature)  # dictionary
        embedded_targets = encode_targets({'target':target_text}, feature)
        query_embedding = embedded_queries['query']
        target_embedding = embedded_targets['target']
        sim_score = (1 - cdist(np.array([query_embedding]), np.array([target_embedding]),
                                   metric=similarity_measure)[0][0])*100

    return sim_score