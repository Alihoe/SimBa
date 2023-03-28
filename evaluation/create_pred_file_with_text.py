import pandas as pd

from evaluation import DATA_PATH
from src.utils import get_queries, get_targets


def create_pred_file_with_text(data_name_orig, data_name, data_name_targets, score_threshold=25):

    queries_path = DATA_PATH + data_name_orig + "/queries.tsv"
    targets_path = DATA_PATH + data_name_targets + "/corpus"
    pred_path = DATA_PATH + data_name + "/pred_qrels.tsv"

    columns = ['query_id', 'target_id', 'score', 'query_text', 'target_text']

    queries = get_queries(queries_path)
    targets = get_targets(targets_path)

    pred_df = pd.read_csv(pred_path, names=['qid', 'Q0', 'docno', 'rank', 'score', 'tag'], sep='\t', index_col=False)

    output_df = pd.DataFrame(columns=columns)

    for _, row in pred_df.iterrows():
        if float(row['score']) >= score_threshold:
            new_row = [row['qid'], row['docno'], row['score'], queries[str(row['qid'])], targets[str(row['docno'])]]
            new_df = pd.DataFrame([new_row], columns=columns)
            output_df = pd.concat([output_df, new_df], names=columns)

    output_path = DATA_PATH + data_name + "/pred_with_text_"+str(score_threshold)+".tsv"
    output_df.to_csv(output_path, index=False, header=True, sep='\t')


numbers = ["11155", "33888", "35529", "44346", "74901", "79409", "76005", "79636", "75199", "77659", "75302", "76981",
           "74465", "72164", "73518", "44346", "79197", "75202", "79639"]

for number in numbers:

    data_name_queries = number + "/" + number + "_pp"
    data_name_cache = number + "_text"
    data_name_targets = 'gesis_unsup_text'
    data_name = number + "/" + number + "_ne_count"

    create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=1)
    create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=2)
    create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=3)




