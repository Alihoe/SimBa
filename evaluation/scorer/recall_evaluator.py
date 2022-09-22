import pdb
import logging
import argparse
import os
from zipfile import ZipFile

import pandas as pd
from trectools import TrecRun, TrecQrel, TrecEval
from os.path import join, dirname, abspath
import pickle as pkl

import sys

from evaluation.format_checker.main import check_format
from evaluation.scorer import DATA_PATH
from evaluation.scorer.utils import print_single_metric, print_thresholded_metric
from src.utils import load_pickled_object, decompress_file, output_dict_to_pred_qrels

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

    rel = results.get_relevant_documents()
    rel_ret = results.get_relevant_retrieved_documents()
    recall = rel_ret/rel

    print('Number of relevant documents:' + str(rel))
    print('Number of retrieved relevant documents:' + str(rel_ret))

    return recall


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

# "../../data/clef_2022_checkthat_2a_english/pred_qrels.tsv"
# "../../data/clef_2022_checkthat_2a_english/gold.tsv"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2022_checkthat_2a_english")
    parser.add_argument('--after_re_ranking', action="store_true") #otherwise measure recall after retrieval
    args = parser.parse_args()

    if args.after_re_ranking:
        pred_file = DATA_PATH + args.data + "/pred_qrels.tsv"
    else:
        candidates_path = DATA_PATH + args.data + "/candidates"
        candidates = load_pickled_object(decompress_file(candidates_path + ".pickle" + ".zip"))
        output_dict_to_pred_qrels(candidates, DATA_PATH+args.data+"/candidates.tsv")
        pred_file = DATA_PATH+args.data+"/candidates.tsv"

    line_separator = '=' * 120

    gold_file = DATA_PATH + args.data + "/gold.tsv"

    if validate_files(pred_file, gold_file):
        results = evaluate(gold_file, pred_file)
        print_single_metric('RECALL:', results)
    if os.path.exists(DATA_PATH+args.data+"/candidates.tsv"):
        os.remove(DATA_PATH+args.data+"/candidates.tsv")

