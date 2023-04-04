import argparse
import os
import sys

import pandas as pd

base_path = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(base_path, "../data/")
sys.path.append(os.path.join(base_path, "../../src"))
from src.utils import get_queries, get_targets

def run():

    parser = argparse.ArgumentParser()
    #input
    parser.add_argument('data_name_queries', type=str)
    parser.add_argument('data_name_targets', type=str)
    parser.add_argument('data_name', type=str)
    parser.add_argument('score_threshold', type=int)
    args = parser.parse_args()

    queries_path = args.data_name_queries
    targets_path = args.data_name_targets
    pred_path = DATA_PATH + args.data_name + "/pred_qrels.tsv"

    columns = ['query_id', 'target_id', 'score', 'query_text', 'target_text']

    queries = get_queries(queries_path)
    targets = get_targets(targets_path)

    pred_df = pd.read_csv(pred_path, names=['qid', 'Q0', 'docno', 'rank', 'score', 'tag'], sep='\t', index_col=False)

    output_df = pd.DataFrame(columns=columns)

    for _, row in pred_df.iterrows():
        if float(row['score']) >= args.score_threshold:
            new_row = [row['qid'], row['docno'], row['score'], queries[str(row['qid'])], targets[str(row['docno'])]]
            new_df = pd.DataFrame([new_row], columns=columns)
            output_df = pd.concat([output_df, new_df], names=columns)

    output_path = DATA_PATH + args.data_name + "/pred_with_text_"+str(args.score_threshold)+".tsv"
    output_df.to_csv(output_path, index=False, header=True, sep='\t')


if __name__ == "__main__":
    run()

#
# numbers = ["11155", "44346", "79409", "75302", "79639"]
#
# for number in numbers:
#
#     data_name_queries = number + "/" + number + "_pp"
#     data_name_cache = number + "_labels"
#     data_name_targets = 'gesis_unsup_labels'
#     data_name = number + "/" + number + "_spacy_ne_count_no_nr_text"
#
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=1)
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=2)
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=3)
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=4)
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=5)
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=6)
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=7)
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=8)
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=9)
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=10)


