import pandas as pd

from evaluation import DATA_PATH


def keep_n_lines(input_path, output_path, n):
    original = pd.read_csv(input_path, sep='\t', index_col=False)
    number_of_rows = len(original.index)
    x = int(number_of_rows/n)
    altered = original.iloc[::x, :]
    altered.to_csv(output_path, index=False, header=True, sep='\t')


def make_gold_and_queries_file(input_path_targets, input_path_queries, query_output_path, gold_output_path):
    original_targets = pd.read_csv(input_path_targets, sep='\t', index_col=False)
    print(original_targets)
    original_queries = pd.read_csv(input_path_queries, sep='\t', index_col=False)
    print(original_queries)
    gold_columns = ["query", "0", "target", "1"]
    query_columns = ["query", "text"]
    gold_df = pd.DataFrame(columns=gold_columns)
    query_df = pd.DataFrame(columns=query_columns)
    query_df["query"] = original_queries["query"]
    query_df["text"] = original_queries["text"]
    gold_df["query"] = original_queries["query"]
    gold_df["target"] = original_targets["id"]
    gold_df["0"] = 0
    gold_df["1"] = 1

    query_df.to_csv(query_output_path, index=False, header=False, sep='\t')
    gold_df.to_csv(gold_output_path, index=False, header=False, sep='\t')

#keep_n_lines(DATA_PATH + "gesis_unsup_more_text/corpus", DATA_PATH + "gesis_test", 20)
make_gold_and_queries_file(DATA_PATH + "gesis_test", DATA_PATH + "gesis_test_answers", DATA_PATH + "gesis_test_sup/queries.tsv", DATA_PATH + "gesis_test_sup/gold.tsv")