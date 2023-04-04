import logging
import argparse
import os

import pandas as pd
from trectools import TrecRun, TrecQrel, TrecEval


import sys

base_path = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(base_path, "../../data/")
sys.path.append(os.path.join(base_path, "../../evaluation/format_checker"))
sys.path.append(os.path.join(base_path, "../../evaluation/scorer"))
sys.path.append(os.path.join(base_path, "../../src"))

#from claimlinking_simba.evaluation import DATA_PATH
from main import check_format
from scorer_utils import print_single_metric, print_thresholded_metric
from utils import load_pickled_object, decompress_file, candidates_dict_to_pred_qrels


# from evaluation.format_checker.main import check_format
# from evaluation.scorer import DATA_PATH
# from evaluation.scorer.utils import print_single_metric
# from src.utils import load_pickled_object, decompress_file, candidates_dict_to_pred_qrels

sys.path.append('.')
"""
Scoring of Task 2 with the metrics Average Precision, R-Precision, P@N, RR@N. 
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 50, 1000]


def evaluate(gold_fpath, pred_fpath, thresholds=None):
    """
    Evaluates the predicted line rankings w.r.t. a gold file.
    Metrics are: Average Precision, R-Pr, Reciprocal Rank, Precision@N
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number at each line, where the list is ordered by check-worthiness.
    :param thresholds: thresholds used for Reciprocal Rank@N and Precision@N.
    If not specified - 1, 3, 5, 10, 20, 50, len(ranked_lines).
    """
    gold_labels = TrecQrel(gold_fpath)
    prediction = TrecRun(pred_fpath)
    results = TrecEval(prediction, gold_labels)

    ndcg = [results.get_ndcg(depth=i) for i in MAIN_THRESHOLDS]
    return ndcg


def validate_files(pred_file, gold_file):
    if not check_format(pred_file):
        logging.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
        return False

    # Checking that all the input tweets are in the prediciton file and have predicitons. 
    pred_names = ['iclaim_id', 'zero', 'vclaim_id', 'rank', 'score', 'tag']
    pred_df = pd.read_csv(pred_file, sep='\t', names=pred_names, index_col=False)
    gold_names = ['iclaim_id', 'zero', 'vclaim_id', 'relevance']
    gold_df = pd.read_csv(gold_file, sep='\t', names=gold_names, index_col=False)
    for iclaim in set(gold_df.iclaim_id):
        if iclaim not in pred_df.iclaim_id.tolist():
            logging.error('Missing iclaim {}. Cannot score.'.format(iclaim))
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gold', type=str, help='Path to goldfile.')
    parser.add_argument('pred', type=str, help='Path to predfile.')
    args = parser.parse_args()

    pred_file = args.pred
    gold_file = args.gold

    if validate_files(pred_file, gold_file):
        results = evaluate(gold_file, pred_file)
        print_thresholded_metric('NDCG@N:', MAIN_THRESHOLDS, results)


