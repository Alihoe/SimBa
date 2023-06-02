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

def create_pred_file_with_text_stop_words(data_name_orig, data_name, data_name_targets_stop_words, data_name_targets, score_threshold=25):

    queries_path = DATA_PATH + data_name_orig + "/queries.tsv"
    targets_path_stop_words = DATA_PATH + data_name_targets_stop_words + "/corpus"
    targets_path = DATA_PATH + data_name_targets + "/corpus"
    pred_path = DATA_PATH + data_name + "/pred_qrels.tsv"

    columns = ['query_id', 'target_id', 'score', 'query_text', 'target_text_stop_words', 'marker', 'target_text']

    queries = get_queries(queries_path)
    targets_stop_words = get_targets(targets_path_stop_words)
    targets = get_targets(targets_path)

    pred_df = pd.read_csv(pred_path, names=['qid', 'Q0', 'docno', 'rank', 'score', 'tag'], sep='\t', index_col=False)

    output_df = pd.DataFrame(columns=columns)

    for _, row in pred_df.iterrows():
        if float(row['score']) >= score_threshold:
            new_row = [row['qid'], row['docno'], row['score'], queries[str(row['qid'])], targets_stop_words[str(row['docno'])], '||||||', targets[str(row['docno'])]]
            new_df = pd.DataFrame([new_row], columns=columns)
            output_df = pd.concat([output_df, new_df], names=columns)

    output_path = DATA_PATH + data_name + "/pred_with_text_"+str(score_threshold)+".tsv"
    output_df.to_csv(output_path, index=False, header=True, sep='\t')

# data_name_queries = "gesis_test_sup"
# data_name_targets = "gesis_unsup_more_text"
# data_name = "gesis_test_sup/gesis_test_sup"+"_more_text_st5"
#
# create_pred_file_with_text(data_name_queries, data_name,  data_name_targets, score_threshold=60)

# number = "gesis_test_sup"
#
# data_name_queries = number
# data_name_cache = number + "_more_text_pp_stop_words_st5"
# data_name_targets_stop_words = 'gesis_unsup_more_text_pp_stop_words'
# data_name_targets = 'gesis_unsup_more_text'
# data_name = number + "/" + number + "_more_text_pp_stop_words_st5"
#
# create_pred_file_with_text_stop_words(data_name_queries, data_name, data_name_targets_stop_words, data_name_targets, score_threshold=72)
#
# data_name_cache = number + "_text_pp_stop_words_st5"
# data_name_targets_stop_words = 'gesis_unsup_text_pp_stop_words'
# data_name_targets = 'gesis_unsup_text'
# data_name = number + "/" + number + "_text_pp_stop_words_st5"
#
# create_pred_file_with_text_stop_words(data_name_queries, data_name, data_name_targets_stop_words, data_name_targets, score_threshold=72)
#
#
# data_name_cache = number + "_labels_title_pp_stop_words_st5"
# data_name_targets_stop_words = 'gesis_unsup_labels_title_pp_stop_words'
# data_name_targets = 'gesis_unsup_labels_title'
# data_name = number + "/" + number + "_labels_title_pp_stop_words_st5"
#
# create_pred_file_with_text_stop_words(data_name_queries, data_name, data_name_targets_stop_words, data_name_targets, score_threshold=72)
#
#
# data_name_cache = number + "_labels_pp_stop_words_st5"
# data_name_targets_stop_words = 'gesis_unsup_labels_pp_stop_words'
# data_name_targets = 'gesis_unsup_labels'
# data_name = number + "/" + number + "_labels_pp_stop_words_st5"
#
# create_pred_file_with_text_stop_words(data_name_queries, data_name, data_name_targets_stop_words, data_name_targets, score_threshold=72)
#
# data_name_cache = number + "_more_text"
# data_name_targets = 'gesis_unsup_more_text'
# data_name = number + "/" + number + "_more_text_st5"
#
# create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=72)
#
# data_name_cache = number + "_text_st5"
# data_name_targets = 'gesis_unsup_text'
# data_name = number + "/" + number + "_text_st5"
#
# create_pred_file_with_text(data_name_queries, data_name,  data_name_targets, score_threshold=72)
#
#
# data_name_cache = number + "_labels_title"
# data_name_targets = 'gesis_unsup_labels_title'
# data_name = number + "/" + number + "_labels_title_st5"
#
# create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=72)
#
#
# data_name_cache = number + "_labels"
# data_name_targets = 'gesis_unsup_labels'
# data_name = number + "/" + number + "_labels_st5"
#
# create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=72)
#


# numbers = ["11155", "44346", "79409", "75302", "79639"]
#
#
# for number in numbers:
#
#     data_name_queries = number + "/" + number + "_pp"
#     data_name_cache = number + "_more_text_pp_stop_words_st5"
#     data_name_targets_stop_words = 'gesis_unsup_more_text_pp_stop_words'
#     data_name_targets = 'gesis_unsup_more_text'
#     data_name = number + "/" + number + "_more_text_pp_stop_words_st5"
#
#     create_pred_file_with_text_stop_words(data_name_queries, data_name, data_name_targets_stop_words, data_name_targets, score_threshold=72)
#
#     data_name_cache = number + "_text_pp_stop_words_st5"
#     data_name_targets_stop_words = 'gesis_unsup_text_pp_stop_words'
#     data_name_targets = 'gesis_unsup_text'
#     data_name = number + "/" + number + "_text_pp_stop_words_st5"
#
#     create_pred_file_with_text_stop_words(data_name_queries, data_name, data_name_targets_stop_words, data_name_targets, score_threshold=72)
#
#
#     data_name_cache = number + "_labels_title_pp_stop_words_st5"
#     data_name_targets_stop_words = 'gesis_unsup_labels_title_pp_stop_words'
#     data_name_targets = 'gesis_unsup_labels_title'
#     data_name = number + "/" + number + "_labels_title_pp_stop_words_st5"
#
#     create_pred_file_with_text_stop_words(data_name_queries, data_name, data_name_targets_stop_words, data_name_targets, score_threshold=72)
#
#
#     data_name_cache = number + "_labels_pp_stop_words_st5"
#     data_name_targets_stop_words = 'gesis_unsup_labels_pp_stop_words'
#     data_name_targets = 'gesis_unsup_labels'
#     data_name = number + "/" + number + "_labels_pp_stop_words_st5"
#
#     create_pred_file_with_text_stop_words(data_name_queries, data_name, data_name_targets_stop_words, data_name_targets, score_threshold=72)
#
#     data_name_queries = number + "/" + number + "_pp"
#     data_name_cache = number + "_more_text"
#     data_name_targets = 'gesis_unsup_more_text'
#     data_name = number + "/" + number + "_more_text_st5"
#
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=72)
#
#     data_name_cache = number + "_text_st5"
#     data_name_targets = 'gesis_unsup_text'
#     data_name = number + "/" + number + "_text_st5"
#
#     create_pred_file_with_text(data_name_queries, data_name,  data_name_targets, score_threshold=72)
#
#
#     data_name_cache = number + "_labels_title"
#     data_name_targets = 'gesis_unsup_labels_title'
#     data_name = number + "/" + number + "_labels_title_st5"
#
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=72)
#
#
#     data_name_cache = number + "_labels"
#     data_name_targets = 'gesis_unsup_labels'
#     data_name = number + "/" + number + "_labels_st5"
#
#     create_pred_file_with_text(data_name_queries, data_name, data_name_targets, score_threshold=72)
#


