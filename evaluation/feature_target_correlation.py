from torch import cdist

from evaluation import DATA_PATH
from src.re_ranking.lexical_similarity import get_lexical_similarity_ratio
from src.sentence_encoder import encode_queries, encode_targets
from src.utils import get_queries, get_targets, get_correct_targets, get_predicted_queries_and_targets_df
import pandas as pd
import numpy as np


def create_feature_target_correlation_file(data, sentence_embedding_models, similarity_measure, lexical_similarity_measures, fields):
    pred_path = DATA_PATH + data+ "/pred_qrels.tsv"
    gold_path = DATA_PATH + data+ "/gold.tsv"
    query_path = DATA_PATH + data +"/queries.tsv"
    corpus_path = DATA_PATH+ data +"/corpus"
    text_data_analysis_path = DATA_PATH + "evaluation/" + data + "text_data_analysis.tsv"

    queries = get_queries(query_path)
    targets = get_targets(corpus_path, fields)
    correct_targets = get_correct_targets(gold_path)
    pred_df = get_predicted_queries_and_targets_df(pred_path)

    pred_query_ids = pred_df['query'].tolist()
    pred_target_ids = pred_df['target'].tolist()

    candidate_targets = {k: targets[k] for k in pred_target_ids if k in targets}
    candidate_queries_and_targets = pred_df.to_dict(list)
    #
    all_sim_scores = {}
    all_features = []

    for model in sentence_embedding_models:
        all_features.append(model)

        embedded_queries = encode_queries(queries, model) # dictionary
        relevant_embedded_targets = encode_targets(candidate_targets, model)

        for query_id in pred_query_ids:
            query_embedding = embedded_queries[query_id]
            target_ids = list(candidate_targets.keys())
            target_embeddings = [relevant_embedded_targets[x] for x in target_ids]
            sim_scores = 1 - cdist(np.array([query_embedding]), np.stack(target_embeddings, axis=0), metric=similarity_measure)
            all_sim_scores[query_id].append(sim_scores)

    if "similar_words_ratio" in lexical_similarity_measures:
        all_features.append("similar_words_ratio")
        lexical_similarities = get_lexical_similarity_ratio(queries, candidate_queries_and_targets)
        for query_id, target_sim_scores in list(lexical_similarities.items()):
            sim_scores = list(target_sim_scores.values())
            all_sim_scores[query_id].append(sim_scores)

    text_data_analysis_df = pd.DataFrame(columns=['query', 'target', 'correct_pair', 'sim_scores'])
    for idx, query_id in range(len(pred_query_ids)):
        print(idx)
        print(query_id)
        query = queries[query_id]
        target_id = pred_target_ids[idx]
        target = targets[target_id]
        sim_scores = all_sim_scores[query_id]
        print(query)
        print(target)
        correct_target = correct_targets[query_id]
        print(correct_target)
        if type(correct_target) == list and target_id in correct_target:
            correct_pair = True
        else:
            if correct_target == target_id:
                correct_pair = True
            else:
                correct_pair = False
        print(correct_pair)
        this_row_df = pd.DataFrame([query, target, correct_pair, sim_scores])
        text_data_analysis_df.append(this_row_df)
    text_data_analysis_df.to_csv(text_data_analysis_path, index=False, header=False, sep='\t')










