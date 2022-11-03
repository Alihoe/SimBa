import argparse

from src.analysis import DATA_PATH
from src.utils import get_queries, get_targets, get_correct_targets


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2022_checkthat_2a_english")

    args = parser.parse_args()

    query_path = DATA_PATH+args.data+"/queries.tsv"
    corpus_path = DATA_PATH+args.data+"/corpus"
    gold_path = DATA_PATH+args.data+"/gold.tsv"
    queries = get_queries(query_path) # queries dictionary
    targets = get_targets(corpus_path, args.fields) # targets dictionary
    correct_targets = get_correct_targets(gold_path)
    correct_target_ids = correct_targets.values()

    ### just got here

    ## make dictionary: query: target1: query_text, target_text, target2
    # or
    ## make dataframe:
    # query, target1, querytext, target1_text, sim score 1, sim score 2, sim score 3
    # + random matches

    query_ids = list(queries.keys())
    target_ids = list(targets.keys())

    if args.pre_processing:
        caching_directory = DATA_PATH + "pre_processed_data/cache/" + args.data
    else:
        caching_directory = DATA_PATH + "cache/" + args.data

    if args.fields != 'all':
        fields = '_'.join(args.fields)
        caching_directory = caching_directory + '_' + fields

    if not args.no_cache:
        if not os.path.isdir(caching_directory):
            os.makedirs(caching_directory)

    if args.pre_processing:
        pre_process_data_path = DATA_PATH + "pre_processed_data/" + args.data
        if args.fields != 'all':
            pre_process_data_path = pre_process_data_path + '_' + fields
        queries, targets = pre_process(queries, targets, args.data)
        if not os.path.isdir(pre_process_data_path):
            os.makedirs(pre_process_data_path)
        pickle_object(pre_process_data_path + "/pp_queries", queries)
        compress_file(pre_process_data_path + "/pp_queries" + ".pickle")
        os.remove(pre_process_data_path + "/pp_queries" + ".pickle")
        pickle_object(pre_process_data_path + "/pp_targets", targets)
        compress_file(pre_process_data_path + "/pp_targets" + ".pickle")
        os.remove(pre_process_data_path + "/pp_targets" + ".pickle")

    all_sim_scores = []
    all_features = []

    if args.union_of_top_k_per_feature:
        union_of_top_k_per_feature = {}

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
            embedded_queries = encode_queries(queries, model)
            if not args.no_cache:
                pickle_object(stored_embedded_queries, embedded_queries)
                compress_file(stored_embedded_queries + ".pickle")
                os.remove(stored_embedded_queries + ".pickle")
        if os.path.exists(stored_embedded_targets + ".pickle" + ".zip"):
            embedded_targets = load_pickled_object(decompress_file(stored_embedded_targets+".pickle"+".zip"))
        else:
            embedded_targets = encode_targets(targets, model)
            if not args.no_cache:
                pickle_object(stored_embedded_targets, embedded_targets)
                compress_file(stored_embedded_targets + ".pickle")
                os.remove(stored_embedded_targets + ".pickle")
        sim_scores = 1 - cdist(np.stack(list(embedded_queries.values()), axis=0), np.stack(list(embedded_targets.values()), axis=0), metric=args.similarity_measure)
        all_sim_scores.append(sim_scores)

        analyse_correlation(all_features, np.array(all_sim_scores), args.correlation, args.data)

if __name__ == "__main__":
    run()