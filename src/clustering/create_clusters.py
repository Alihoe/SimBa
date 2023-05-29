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


from utils import get_queries, get_targets, all_targets_as_query_candidates, load_pickled_object, \
    decompress_file, get_candidate_targets, pickle_object, compress_file, supervised_output_to_pred_qrels, \
    output_dict_to_pred_qrels
from sentence_encoder import encode_queries, encode_targets
from referential_similarity import get_sequence_entities
from string_similarity import get_string_similarity
from lexical_similarity import get_lexical_entities
from create_feature_set import create_feature_set, create_test_set


def run():
    """
    input:
    queries, targets, {query: list of top k targets (ordered if union is not chosen)}
    output:
    {query: list of top k targets (ordered if union is not chosen)}

    all_sim_scores: {query_id: list_of_sim_scores, list entries are arrays of shape (1, target_n) with target_n similarity scores between 0 and 100}
    """
    parser = argparse.ArgumentParser()
    #input
    parser.add_argument('targets', type=str, help='Input targets path as tsv file.')
    # parameters
    parser.add_argument('data', type=str, help='Name under which the documents should be stored.')
    parser.add_argument('n_clusters', type=int, default=10, help='Number of clusters if necessary')
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
    targets_path = os.path.join(DATA_PATH, args.targets + "/corpus")
    caching_directory_targets = os.path.join(DATA_PATH, "cache", args.targets)
    targets = get_targets(targets_path)

    all_features = []
    output_path = os.path.join(DATA_PATH, args.data)
    print(output_path)
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
        stored_embedded_targets = caching_directory_targets + "/embedded_targets_" + model_name
        if os.path.exists(stored_embedded_targets + ".pickle" + ".zip"):
            embedded_targets = load_pickled_object(decompress_file(stored_embedded_targets+".pickle"+".zip"))
            print('targets loaded')
        else:
            print('compute targets')
            embedded_targets = encode_targets(targets, model)
            pickle_object(stored_embedded_targets, embedded_targets)
            compress_file(stored_embedded_targets + ".pickle")
            os.remove(stored_embedded_targets + ".pickle")
    """
    2. For all referential similarity measures\
    2.1 get entities for all queries and cache or load from cache\
    2.2. get entities for all targets and cache or load from cache\
    2.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
    """
    for ref_feature in args.referential_similarity_measures:
        print(ref_feature)
        all_features.append(ref_feature)
        stored_entities_targets = caching_directory_targets + "/targets_" + str(ref_feature)
        if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
            entities_targets = load_pickled_object(decompress_file(stored_entities_targets + ".pickle" + ".zip"))
            print('targets loaded')
        else:
            print('compute targets')
            entities_targets = get_sequence_entities(targets, ref_feature)
            pickle_object(stored_entities_targets, entities_targets)
            compress_file(stored_entities_targets + ".pickle")
            os.remove(stored_entities_targets + ".pickle")
    """
    3. For all lexical similarity measures
    3.1 get entities for all queries and cache or load from cache\
    3.2. get entities for all targets and cache or load from cache\
    3.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    """
    for lex_feature in args.lexical_similarity_measures:
        print(lex_feature)
        all_features.append(lex_feature)
        stored_entities_targets = caching_directory_targets + "/targets_" + str(lex_feature)
        if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
            entities_targets = load_pickled_object(decompress_file(stored_entities_targets + ".pickle" + ".zip"))
            print('targets loaded')
        else:
            print('compute targets')
            entities_targets = get_lexical_entities(targets, lex_feature)
            pickle_object(stored_entities_targets, entities_targets)
            compress_file(stored_entities_targets + ".pickle")
            os.remove(stored_entities_targets + ".pickle")

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, n_init="auto").fit(np.array(list(embedded_targets.values())))
    classifer_output_path = output_path + "/classifier.pkl"
    with open(classifer_output_path, 'wb') as classifer_output_file:
        pickle.dump(kmeans, classifer_output_file)
    # print(kmeans.labels_)
    # print(len(kmeans.labels_))
    # all_clusters = {}
    # for cluster in range(args.n_clusters):
    #     all_clusters[cluster] = {}
    # for i in range(len(targets)):
    #     all_clusters[kmeans.labels_[i]][list(targets.keys())[i]] = list(targets.values())[i]
    # for cluster in range(args.n_clusters):
    #     this_cluster_df = pd.DataFrame(columns=['id', 'text'])
    #     this_cluster_df['id'] = list(all_clusters[cluster].keys())
    #     this_cluster_df['text'] = list(all_clusters[cluster].values())
    #     print(cluster)
    #     print(len(list(all_clusters[cluster].values())))
    #     print(list(all_clusters[cluster].values())[:20])
    #     this_output_path = output_path + "/cluster_"+str(cluster+1)+".tsv"
    #     print(this_output_path)
    #     this_cluster_df.to_csv(this_output_path, sep='\t', header=True, index=False)


if __name__ == "__main__":
    run()
