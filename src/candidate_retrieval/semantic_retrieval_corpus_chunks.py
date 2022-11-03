import argparse
import os
from pathlib import Path
import random

import torch
from scipy.spatial.distance import braycurtis, cdist
import numpy as np

from src.candidate_retrieval import DATA_PATH
from src.analysis.correlation_analysis import analyse_correlation
from src.pre_processing.pre_process import pre_process
from src.sentence_encoder import encode_queries, encode_targets
from src.utils import get_queries, get_targets, pickle_object, compress_file, decompress_file, load_pickled_object, \
    make_top_k_dictionary

# parameters
# clef_2021_checkthat_2a_english braycurtis spearman 50


# possible data names:
# "clef_2022_checkthat_2a_english"
# "clef_2022_checkthat_2b_english"
# "clef_2020_checkthat_2_english"
# "clef_2021_checkthat_2a_english"

# example sentence embedding models
# ["all-mpnet-base-v2",
# "princeton-nlp/sup-simcse-roberta-large",
# "sentence-transformers/sentence-t5-base",
# "infersent",
# "https://tfhub.dev/google/universal-sentence-encoder/4"]

# possible similarity measures:
# "braycurtis"
# "canberra"
# "chebyshev"
# "cityblock"
# "correlation"
# "cosine"
# "euclidean"
# "jensenshannon"
# "mahalanobis"
# "minkowski"
# "seuclidean"
# "sqeuclidean"

# possible correlation measures:
# "spearman"
# "mean_squared_errors"

# ["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"],
#

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2020_checkthat_2_english")
    parser.add_argument('corpus_size', type=int, default=1000)
    parser.add_argument('--pre_processing', action='store_true')
    parser.add_argument('similarity_measure', type=str, default='braycurtis')
    parser.add_argument('correlation', type=str, default='spearmanr')
    parser.add_argument('--union_of_top_k_per_feature', action="store_true") # otherwise top k of mean of features
    parser.add_argument('k', type=int, default = 50)
    parser.add_argument('-sentence_embedding_models', type=str, nargs='+',
                    default=["all-mpnet-base-v2"],
                             # "sentence-transformers/sentence-t5-base", "infersent",
                             # "https://tfhub.dev/google/universal-sentence-encoder/4"],
                    help='Pass a list of sentence embedding models hosted by Huggingface or Tensorflow or simply pass "infersent" to use the infersent encoder.')

    args = parser.parse_args()

    corpus_chunks_path = DATA_PATH + "corpus_chunks/" + args.data
    Path(corpus_chunks_path).mkdir(parents=True, exist_ok=True)

    query_path = DATA_PATH+args.data+"/queries.tsv"
    corpus_path = DATA_PATH+args.data+"/corpus"
    queries = get_queries(query_path) # queries dictionary
    query_ids = list(queries.keys())

    stored_target_ids = corpus_chunks_path + "/target_ids"
    target_ids = load_pickled_object(decompress_file(stored_target_ids + ".pickle" + ".zip"))

    all_sim_scores = []

    if args.union_of_top_k_per_feature:
        union_of_top_k_per_feature = {}

    for model in args.sentence_embedding_models:
        if "/" or ":" or "." in str(model):
            model_name = str(model).replace("/", "_").replace(":", "_").replace(".", "_")
        else:
            model_name = str(model)
        stored_embedded_queries = corpus_chunks_path + "/embedded_queries_" + model_name
        stored_embedded_targets = corpus_chunks_path + "/embedded_targets_" + model_name
        if os.path.exists(stored_embedded_queries + ".pickle" + ".zip"):
            embedded_queries = load_pickled_object(decompress_file(stored_embedded_queries+".pickle"+".zip"))
        else:
            embedded_queries = encode_queries(queries, model)
            pickle_object(stored_embedded_queries, embedded_queries)
            compress_file(stored_embedded_queries + ".pickle")
            os.remove(stored_embedded_queries + ".pickle")
        embedded_targets = load_pickled_object(decompress_file(stored_embedded_targets+".pickle"+".zip"))
        sim_scores = 1 - cdist(np.stack(list(embedded_queries.values()), axis=0), np.stack(list(embedded_targets.values()), axis=0), metric=args.similarity_measure)
        all_sim_scores.append(sim_scores)

        if args.union_of_top_k_per_feature:

            sim_scores_top_k_values, sim_scores_top_k_idx = torch.topk(torch.tensor(sim_scores), min(args.k + 1, len(sim_scores[1])),
                                                                       dim=1, largest=True)
            sim_scores_top_k_values = sim_scores_top_k_values.cpu().tolist()
            sim_scores_top_k_idx = sim_scores_top_k_idx.cpu().tolist()

            this_model_top_k = make_top_k_dictionary(query_ids, target_ids, sim_scores_top_k_values, sim_scores_top_k_idx)
            for query_itr in range(len(query_ids)):
                query_id = query_ids[query_itr]
                if query_id in union_of_top_k_per_feature:
                    union_of_top_k_per_feature[query_id] = union_of_top_k_per_feature[query_id] | this_model_top_k[query_id]
                else:
                    union_of_top_k_per_feature[query_id] = this_model_top_k[query_id]

    #analyse_correlation(np.array(all_sim_scores), args.correlation)

    if not args.union_of_top_k_per_feature:
        sim_scores_mean = np.mean(np.array(all_sim_scores), axis=0)

        sim_scores_top_k_values, sim_scores_top_k_idx = torch.topk(torch.tensor(sim_scores_mean),
                                                                   min(args.k + 1, len(sim_scores[1])),
                                                                   dim=1, largest=True)
        sim_scores_top_k_values = sim_scores_top_k_values.cpu().tolist()
        sim_scores_top_k_idx = sim_scores_top_k_idx.cpu().tolist()
        output = make_top_k_dictionary(query_ids, target_ids, sim_scores_top_k_values, sim_scores_top_k_idx)

    else:
        union_of_top_k_per_feature_dict = {}
        for query_id in query_ids:
            union_of_top_k_per_feature_dict[query_id] = dict(sorted(union_of_top_k_per_feature[query_id].items(), key=lambda item:item[1], reverse=True))
        output = union_of_top_k_per_feature_dict

    output_path = corpus_chunks_path+"/candidates"

    pickle_object(output_path, output)
    compress_file(output_path + ".pickle")
    os.remove(output_path + ".pickle")


if __name__ == "__main__":
    run()