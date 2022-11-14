from scipy.spatial.distance import cdist
from evaluation import DATA_PATH
from src.analysis.correlation_analysis import analyse_feature_correlation
from src.re_ranking.lexical_similarity import get_lexical_similarity_ratio
from src.re_ranking.referential_similarity import get_synonym_similarity, get_ne_similarity
from src.re_ranking.string_similarity import get_jacquard_similarity, get_sequence_matching_similarity, \
    get_levenshtein_similarity
from src.sentence_encoder import encode_queries, encode_targets
from src.utils import get_queries, get_targets, get_correct_targets, get_predicted_queries_and_targets_df
import pandas as pd
import numpy as np


def create_feature_target_correlation_file(data, sentence_embedding_models, similarity_measure, lexical_similarity_measures, string_similarity_measures, referential_similarity_measures, fields):
    pred_path = DATA_PATH + data+ "/pred_qrels.tsv"
    gold_path = DATA_PATH + data+ "/gold.tsv"
    query_path = DATA_PATH + data +"/queries.tsv"
    corpus_path = DATA_PATH+ data +"/corpus"
    text_data_analysis_path = DATA_PATH + "evaluation/" + data + "_text_data_analysis.tsv"

    queries = get_queries(query_path)
    targets = get_targets(corpus_path, fields)
    correct_targets = get_correct_targets(gold_path)
    pred_df = get_predicted_queries_and_targets_df(pred_path)

    pred_query_ids_not_unique = pred_df['query'].tolist()
    pred_query_ids = list(set(pred_df['query'].tolist()))
    pred_target_ids = pred_df['target'].tolist()

    candidate_targets = {k: targets[k] for k in pred_target_ids if k in targets}

    candidate_queries_and_targets = {}

    for query_id in pred_query_ids:
        candidate_queries_and_targets[query_id] = {}

    for index, row in pred_df.iterrows():
        query_id = row['query']
        target_id = row['target']
        target_text = targets[target_id]
        candidate_queries_and_targets[query_id][target_id] = target_text

    all_sim_scores = {}
    all_features = []

    for query_id in pred_query_ids:
        all_sim_scores[query_id] = []

    if "similar_words_ratio" in lexical_similarity_measures:
        all_features.append("similar_words_ratio")
        lexical_similarities = get_lexical_similarity_ratio(queries, candidate_queries_and_targets)
        for query_id, target_sim_scores in list(lexical_similarities.items()):
            sim_scores = list(target_sim_scores.values())
            all_sim_scores[query_id].append(sim_scores)

    if "sequence_matching_similarity" in string_similarity_measures:
        all_features.append("sequence_matching_similarity")
        sequence_matching_similarities = get_sequence_matching_similarity(queries, candidate_queries_and_targets)
        for query_id, target_sim_scores in list(sequence_matching_similarities.items()):
            sim_scores = list(target_sim_scores.values())
            all_sim_scores[query_id].append(sim_scores)

    if "levenshtein_similarity" in string_similarity_measures:
        all_features.append("levenshtein_similarity")
        levenshtein_similarities = get_levenshtein_similarity(queries, candidate_queries_and_targets)
        for query_id, target_sim_scores in list(levenshtein_similarities.items()):
            sim_scores = list(target_sim_scores.values())
            all_sim_scores[query_id].append(sim_scores)

    if "jacquard_similarity" in string_similarity_measures:
        all_features.append("jacquard_similarity")
        jacquard_similarities = get_jacquard_similarity(queries, candidate_queries_and_targets)
        for query_id, target_sim_scores in list(jacquard_similarities.items()):
            sim_scores = list(target_sim_scores.values())
            all_sim_scores[query_id].append(sim_scores)

    if "synonym_similarity" in referential_similarity_measures:
        all_features.append("synonym_similarity")
        synonym_similarities = get_synonym_similarity(queries, candidate_queries_and_targets)
        for query_id, target_sim_scores in list(synonym_similarities.items()):
            sim_scores = list(target_sim_scores.values())
            all_sim_scores[query_id].append(sim_scores)

    if "ne_similarity" in referential_similarity_measures:
        all_features.append("ne_similarity")
        ne_similarities = get_ne_similarity(queries, candidate_queries_and_targets)
        for query_id, target_sim_scores in list(ne_similarities.items()):
            sim_scores = list(target_sim_scores.values())
            all_sim_scores[query_id].append(sim_scores)

    for model in sentence_embedding_models:
        all_features.append(model)

        embedded_queries = encode_queries(queries, model) # dictionary
        relevant_embedded_targets = encode_targets(candidate_targets, model)

        for query_id in pred_query_ids:
            query_embedding = embedded_queries[query_id]
            target_ids = list(candidate_queries_and_targets[query_id].keys())
            target_embeddings = [relevant_embedded_targets[x] for x in target_ids]
            sim_scores = 1 - cdist(np.array([query_embedding]), np.stack(target_embeddings, axis=0), metric=similarity_measure)
            all_sim_scores[query_id].append(sim_scores[0])

    columns = ['query_id', 'target_id', 'query', 'target', 'correct_pair']
    columns2 = ['correct_pair']

    for feature in all_features:
        columns2.append(feature)

    text_data_analysis_df = pd.DataFrame(columns=columns)

    corr_analysis = []
    correct_predicted = False

    target_idx = 0
    old_query_id = pred_query_ids_not_unique[0]

    for idx, query_id in enumerate(pred_query_ids_not_unique):
        if query_id != old_query_id:
            if not correct_predicted:
                if type(correct_target) != list:
                    correct_target = [correct_target]
                for c_target in correct_target:
                    c_target_text = targets[c_target]
                    this_row_df = pd.DataFrame([['not predicted', c_target, '-', c_target_text, correct_pair]], columns=columns)
                    text_data_analysis_df = pd.concat([text_data_analysis_df, this_row_df])
            correct_predicted = False
            target_idx = 0
            old_query_id = query_id
        query = queries[query_id]
        target_id = pred_target_ids[idx]
        target = targets[target_id]
        sim_scores = []
        for feature_n in range(len(all_features)):
            sim_scores.append(all_sim_scores[query_id][feature_n][target_idx])
        correct_target = correct_targets[query_id]
        if type(correct_target) == list and target_id in correct_target:
            correct_pair = True
        else:
            if correct_target == target_id:
                correct_pair = True
                correct_predicted = True
            else:
                correct_pair = False
        this_row_df = pd.DataFrame([[query_id, target_id, query, target, correct_pair]], columns=columns)
        text_data_analysis_df = pd.concat([text_data_analysis_df, this_row_df])
        this_row_correlation = [int(correct_pair)]
        for sim_score in sim_scores:
            this_row_correlation.append(sim_score)
        corr_analysis.append(this_row_correlation)
        target_idx = target_idx + 1

    analyse_feature_correlation(columns2, np.array(corr_analysis), 'spearmanr', data)

    text_data_analysis_df.to_csv(text_data_analysis_path, index=False, header=True, sep='\t')

create_feature_target_correlation_file('sv_ident_train_and_val',
                                       ["all-mpnet-base-v2",'Sahajtomar/German-semantic', 'distiluse-base-multilingual-cased-v1'],
                                        'braycurtis',
                                       "similar_words_ratio",
                                       ["sequence_matching_similarity", "levenshtein_similarity", "jacquard_similarity"],
                                       ["ne_similarity", "synonym_similarity"],
                                        "all")

#
# create_feature_target_correlation_file('sv_ident_trial_en',
#                                        ["all-mpnet-base-v2"],#, "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", 'Sahajtomar/German-semantic', 'distiluse-base-multilingual-cased-v1'],
#                                         'cosine',
#                                        "similar_words_ratio",
#                                         "all")








