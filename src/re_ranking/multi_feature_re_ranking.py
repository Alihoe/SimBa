import argparse
import os
import numpy as np

from scipy.spatial.distance import cdist

from src.candidate_retrieval import DATA_PATH
from src.correlation_analysis import analyse_correlation
from src.re_ranking.lexical_similarity import get_lexical_similarity_ratio
from src.re_ranking.ranking_utils import get_queries_and_targets_from_candidates, get_all_relevant_targets
from src.re_ranking.supervised_reranking_methods.utils import prepare_binary_dataset
from src.sentence_encoder import encode_queries, encode_targets
from src.utils import load_pickled_object, decompress_file, pickle_object, compress_file, get_queries, \
    output_dict_to_pred_qrels, get_targets

# possible sentence mebedding models
["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"],

# possible lexical similarity measures
# "similar_words_ratio"


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="test")
    parser.add_argument('--pre_processing', action='store_true')
    parser.add_argument('-fields', type=str, nargs='+', default='all')
    parser.add_argument('-sentence_embedding_models', type=str, nargs='+',
                        default= ["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "https://tfhub.dev/google/universal-sentence-encoder/4"],
 #  "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base"
                        help='Pass a list of sentence embedding models hosted by Huggingface or Tensorflow or simply pass "infersent" to use the infersent encoder.')
    parser.add_argument('similarity_measure', type=str, default='cosine')
    parser.add_argument('-lexical_similarity_measures', type=str, nargs='+', default=["similar_words_ratio"])
    parser.add_argument('correlation', type=str, default='braycurtis')
    parser.add_argument('k', type=int, default=5)
    parser.add_argument('--no_cache', action="store_true", help='If not selected, the encodings of the queries and the targets will be stored as compressed pickle files in the data/cache directory.')
    args = parser.parse_args()

    if args.pre_processing:
        caching_directory = DATA_PATH + "pre_processed_data/cache/" + args.data
    else:
        caching_directory = DATA_PATH + "cache/" + args.data

    if args.fields != 'all':
        fields = '_'.join(args.fields)
        caching_directory = caching_directory + '_' + fields

    if args.pre_processing:
        pre_process_data_path = DATA_PATH + "pre_processed_data/" + args.data
        if args.fields != 'all':
            pre_process_data_path = pre_process_data_path + '_' + fields
        queries = load_pickled_object(decompress_file(pre_process_data_path + "/pp_queries" + ".pickle" + ".zip"))
        candidates_path = DATA_PATH + "pre_processed_data/" + args.data + "/candidates"
    else:
        query_path = DATA_PATH+args.data+"/queries.tsv"
        queries = get_queries(query_path)  # queries dictionary
        corpus_path = DATA_PATH + args.data + "/corpus"
        candidates_path = DATA_PATH + args.data + "/candidates"

    if args.fields != 'all':
        candidates_path = candidates_path + '_' + fields

    query_ids = list(queries.keys())
    candidates = load_pickled_object(decompress_file(candidates_path + ".pickle" + ".zip"))
    candidate_queries_and_targets = get_queries_and_targets_from_candidates(candidates, corpus_path, args.fields)
    candidate_targets = get_all_relevant_targets(candidate_queries_and_targets)

    print(candidate_targets)

    all_sim_scores = {}
    all_features = []

    for query_id in query_ids:
        all_sim_scores[query_id] = []

    for model in args.sentence_embedding_models:
        all_features.append(model)
        if "/" or ":" or "." in str(model):
            model_name = str(model).replace("/", "_").replace(":", "_").replace(".", "_")
        else:
            model_name = str(model)
        stored_embedded_queries = caching_directory + "/embedded_queries_" + model_name
        stored_embedded_targets = caching_directory + "/embedded_targets_" + model_name
        if os.path.exists(stored_embedded_queries + ".pickle" + ".zip"):
            embedded_queries = load_pickled_object(decompress_file(stored_embedded_queries+".pickle"+".zip"))
        else:
            embedded_queries = encode_queries(queries, model) # dictionary
            if not args.no_cache:
                pickle_object(stored_embedded_queries, embedded_queries)
                compress_file(stored_embedded_queries + ".pickle")
                os.remove(stored_embedded_queries + ".pickle")
        if os.path.exists(stored_embedded_targets + ".pickle" + ".zip"):
            relevant_embedded_targets = load_pickled_object(decompress_file(stored_embedded_targets+".pickle"+".zip"))
        else:
            relevant_embedded_targets = encode_targets(candidate_targets, model)
            if not args.no_cache:
                pickle_object(stored_embedded_targets+"_only_candidates", relevant_embedded_targets)
                compress_file(stored_embedded_targets + "_only_candidates.pickle")
                os.remove(stored_embedded_targets + "_only_candidates.pickle")
        for query_id in query_ids:
            print(query_id)
            query_embedding = embedded_queries[query_id]
            target_ids = list(candidate_queries_and_targets[query_id].keys())
            target_embeddings = [relevant_embedded_targets[x] for x in target_ids]
            sim_scores = 1 - cdist(np.array([query_embedding]), np.stack(target_embeddings, axis=0), metric=args.similarity_measure)
            all_sim_scores[query_id].append(sim_scores)

    if "similar_words_ratio" in args.lexical_similarity_measures:
        all_features.append("similar_words_ratio")
        lexical_similarities = get_lexical_similarity_ratio(queries, candidate_queries_and_targets)
        for query_id, target_sim_scores in list(lexical_similarities.items()):
            sim_scores = list(target_sim_scores.values())
            all_sim_scores[query_id].append(sim_scores)

    analyse_correlation(all_features, np.array(all_sim_scores), args.correlation, args.data)

    sim_scores_mean = {}

    for query_id, sim_scores in list(all_sim_scores.items()):
        sim_scores_mean[query_id] = np.mean(sim_scores, axis=0)

    all_target_ids = {}

    output = {}
    for query_id, sim_scores in list(sim_scores_mean.items()):
        target_ids = list(candidate_queries_and_targets[query_id].keys())
        all_target_ids[query_id] = target_ids
        sim_scores_per_query = sim_scores.tolist()[0]
        targets_and_sim_scores = dict(zip(target_ids, sim_scores_per_query))
        targets_and_sim_scores = dict(sorted(targets_and_sim_scores.items(), key=lambda item: item[1], reverse=True))
        targets_and_sim_scores = {x: targets_and_sim_scores[x] for x in list(targets_and_sim_scores)[:args.k]}
        output[query_id] = targets_and_sim_scores

    if args.pre_processing:
        output_path = DATA_PATH + "pre_processed_data/" + args.data +"/pred_qrels.tsv"
    else:
        output_path = DATA_PATH+args.data+"/pred_qrels.tsv"

    if args.fields != 'all':
        output_path = output_path + '_' + fields

    output_dict_to_pred_qrels(output, output_path)

    prepare_binary_dataset(all_sim_scores, all_target_ids, DATA_PATH+args.data+"/gold.tsv")


if __name__ == "__main__":
    run()