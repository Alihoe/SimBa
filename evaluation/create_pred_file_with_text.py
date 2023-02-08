import pandas as pd

from evaluation import DATA_PATH
from src.utils import get_queries, get_targets


def create_pred_file_with_text(data_name_orig, data_name, score_threshold=27):

    queries_path = DATA_PATH + data_name_orig + "/queries.tsv"
    targets_path = DATA_PATH + data_name + "/pp_targets.tsv"
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

    output_path = DATA_PATH + data_name + "/pred_with_text.tsv"
    output_df.to_csv(output_path, index=False, header=True, sep='\t')


data_name_orig = '11235'
data_name = '11235_fields_5'


create_pred_file_with_text(data_name_orig, data_name)

