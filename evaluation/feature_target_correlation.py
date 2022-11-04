from evaluation import DATA_PATH
from src.utils import get_queries, get_targets, get_correct_targets, get_predicted_targets


def create_feature_target_correlation_file(data, sentence_embedding_models, lexical_features, fields):
    pred_path = DATA_PATH + data+ "/pred_qrels.tsv"
    gold_path = DATA_PATH + data+ "/gold.tsv"
    query_path = DATA_PATH + data +"/queries.tsv"
    corpus_path = DATA_PATH+ data +"/corpus"

    queries = get_queries(query_path)
    targets = get_targets(corpus_path, fields)
    correct_targets = get_correct_targets(gold_path)
    pred_targets = get_predicted_targets(gold_path)






