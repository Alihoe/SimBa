import argparse
import os
import pickle
import sys

import numpy as np

import tensorflow as tf
import pandas as pd
from sklearn import svm, naive_bayes, preprocessing
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest

from tensorflow.python.framework.ops import EagerTensor

from scipy.spatial.distance import cdist
from pathlib import Path

base_path = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(base_path, "../../data/")
sys.path.append(os.path.join(base_path, "../create_similarity_features"))
sys.path.append(os.path.join(base_path, "../learning"))
sys.path.append(os.path.join(base_path, ".."))
import re_ranking
import lexical_similarity
import referential_similarity
import sentence_encoder
import string_similarity
sys.path.insert(0, os.path.join(base_path, ".."))
import utils
import create_feature_set


from utils import get_queries, get_queries, load_pickled_object, \
    decompress_file, pickle_object, compress_file, \
    output_dict_to_pred_qrels
from sentence_encoder import encode_queries, encode_queries
from referential_similarity import get_sequence_entities
from string_similarity import get_string_similarity
from lexical_similarity import get_lexical_entities
from create_feature_set import create_feature_set, create_test_set


def run():
    """
     """
    parser = argparse.ArgumentParser()
    #input
    parser.add_argument('queries', type=str, help='Input queries path as tsv file.')
    # parameters
    parser.add_argument('data', type=str, help='Name under which the documents should be stored.')
    parser.add_argument('-sentence_embedding_models', type=str, nargs='+',
                    default=[],
                    help='Pass a list of sentence embedding models hosted by Huggingface or Tensorflow or simply pass "infersent" to use the infersent encoder.')
    parser.add_argument('-referential_similarity_measures', type=str, nargs='+',
                        default=[])
    parser.add_argument('-lexical_similarity_measures', type=str, nargs='+', default=[],
                        help='Pass a list of lexical similarity measures to use')
    args = parser.parse_args()
    """
    """
    queries_path = os.path.join(DATA_PATH, args.queries + "/queries.tsv")
    caching_directory_queries = os.path.join(DATA_PATH, "cache", args.queries)
    Path(caching_directory_queries).mkdir(parents=True, exist_ok=True)
    queries = get_queries(queries_path)

    all_features = []
    output_path = os.path.join(DATA_PATH, args.data)
    Path(os.path.join(DATA_PATH, args.data)).mkdir(parents=True, exist_ok=True)
    """
    all sentence embedding models
     """
    for model in args.sentence_embedding_models:
        print(model)
        all_features.append(model)
        if "/" or ":" or "." in str(model):
            model_name = str(model).replace("/", "_").replace(":", "_").replace(".", "_")
        else:
            model_name = str(model)
        stored_embedded_queries = caching_directory_queries + "/embedded_queries_" + model_name
        if os.path.exists(stored_embedded_queries + ".pickle" + ".zip"):
            embedded_queries = load_pickled_object(decompress_file(stored_embedded_queries+".pickle"+".zip"))
            print('queries loaded')
        else:
            print('compute queries')
            embedded_queries = encode_queries(queries, model)
            pickle_object(stored_embedded_queries, embedded_queries)
            compress_file(stored_embedded_queries + ".pickle")
            os.remove(stored_embedded_queries + ".pickle")
    """
    2. For all referential similarity measures\
    2.1 get entities for all queries and cache or load from cache\
    2.2. get entities for all queries and cache or load from cache\
    2.3. Calculate all similarity scores for one query and its *candidate queries* or load from cache -> value between 0 and 100 and cache
    """
    for ref_feature in args.referential_similarity_measures:
        print(ref_feature)
        all_features.append(ref_feature)
        stored_entities_queries = caching_directory_queries + "/queries_" + str(ref_feature)
        if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
            entities_queries = load_pickled_object(decompress_file(stored_entities_queries + ".pickle" + ".zip"))
            print('queries loaded')
        else:
            print('compute queries')
            entities_queries = get_sequence_entities(queries, ref_feature)
            pickle_object(stored_entities_queries, entities_queries)
            compress_file(stored_entities_queries + ".pickle")
            os.remove(stored_entities_queries + ".pickle")
    """
    3. For all lexical similarity measures
    3.1 get entities for all queries and cache or load from cache\
    3.2. get entities for all queries and cache or load from cache\
    3.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    """
    for lex_feature in args.lexical_similarity_measures:
        print(lex_feature)
        all_features.append(lex_feature)
        stored_entities_queries = caching_directory_queries + "/queries_" + str(lex_feature)
        if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
            entities_queries = load_pickled_object(decompress_file(stored_entities_queries + ".pickle" + ".zip"))
            print('queries loaded')
        else:
            print('compute queries')
            entities_queries = get_lexical_entities(queries, lex_feature)
            pickle_object(stored_entities_queries, entities_queries)
            compress_file(stored_entities_queries + ".pickle")
            os.remove(stored_entities_queries + ".pickle")

    classifer_output_path = output_path + "/classifier.pkl"
    with open(classifer_output_path, 'rb') as classifer_output_file:
        kmeans = pickle.load(classifer_output_file)
    predictions = kmeans.predict(np.array(list(embedded_queries.values())))
    queries_clusters = dict(zip(list(queries.keys()), list(predictions)))

    #cluster_themes = ["media", "politics", "feelings", "personal_inf", "job"]


    all_target_ids = []
    print(kmeans.n_clusters)
    for cluster in range(kmeans.n_clusters):
        these_targets_path = output_path + "/cluster_"+str(cluster+1)+".tsv"
        these_targets = pd.read_csv(these_targets_path, sep='\t', dtype=str)
        all_target_ids.append(these_targets['id'].to_list())

    print(output_path)
    output_path = output_path+"/candidates"

    output = {}
    for idx, query_id in enumerate(list(queries.keys())):
        cluster_nr = predictions[idx]
        output[query_id] = all_target_ids[cluster_nr]



    pickle_object(output_path, output)
    compress_file(output_path + ".pickle")
    os.remove(output_path + ".pickle")


if __name__ == "__main__":
    run()
