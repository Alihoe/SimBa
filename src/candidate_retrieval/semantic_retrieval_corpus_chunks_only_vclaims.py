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

    corpus_path = DATA_PATH+args.data+"/corpus"
    targets = get_targets(corpus_path) # targets dictionary

    org_corp_size = len(list(targets.keys()))
    pop_nr = org_corp_size - args.corpus_size
    random.seed(2)
    for i in range(pop_nr):
        targets.pop(random.choice(list(targets.keys())))

    stored_target_ids = corpus_chunks_path + "/target_ids"

    target_ids = list(targets.keys())
    pickle_object(stored_target_ids, target_ids)
    compress_file(stored_target_ids + ".pickle")
    os.remove(stored_target_ids + ".pickle")

    for model in args.sentence_embedding_models:
        if "/" or ":" or "." in str(model):
            model_name = str(model).replace("/", "_").replace(":", "_").replace(".", "_")
        else:
            model_name = str(model)
        stored_embedded_targets = corpus_chunks_path + "/embedded_targets_" + model_name
        embedded_targets = encode_targets(targets, model)
        pickle_object(stored_embedded_targets, embedded_targets)
        compress_file(stored_embedded_targets + ".pickle")
        os.remove(stored_embedded_targets + ".pickle")



if __name__ == "__main__":
    run()