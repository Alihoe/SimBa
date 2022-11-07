import argparse
import subprocess

from src import DATA_PATH
from src.utils import append_qrels_documents


def run():

    english_data = 'sv_ident_en_train_and_val'
    german_data = 'sv_ident_de_train_and_val'

    subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval.py", english_data, "cosine", "--union_of_top_k_per_feature", "spearmanr", "50", '-sentence_embedding_models', 'sentence-transformers/sentence-t5-base', '-fields', 'variable_label', 'topic_en', 'question_text', 'question_text_en'])
    subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking.py", english_data, "cosine", "spearmanr", "10", '-sentence_embedding_models', 'sentence-transformers/sentence-t5-base', '-fields', 'variable_label', 'topic_en', 'question_text', 'question_text_en', '-lexical_similarity_measures=False'])

    subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval.py", german_data, "cosine", "--union_of_top_k_per_feature", "spearmanr", "50", '-sentence_embedding_models', 'Sahajtomar/German-semantic', '-fields', 'study_title', 'variable_label', 'variable_name', 'answer_categories', 'topic', 'question_text', 'sub_question', 'item_categories'])
    subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking.py", german_data, "cosine", "spearmanr", "10", '-sentence_embedding_models', 'Sahajtomar/German-semantic','-fields', 'study_title', 'variable_label', 'variable_name', 'answer_categories', 'topic', 'question_text', 'sub_question', 'item_categories', '-lexical_similarity_measures=False'])

    pred_qrels_english = DATA_PATH + english_data + "/pred_qrels.tsv" + '_variable_label_topic_en_question_text_question_text_en'
    pred_qrels_german = DATA_PATH + german_data + "/pred_qrels.tsv" + '_study_title_variable_label_variable_name_answer_categories_topic_question_text_sub_question_item_categories'

    pred_file = DATA_PATH + german_data + '_' + english_data + 'pred_qrels.tsv'

    append_qrels_documents(pred_qrels_english, pred_qrels_german, pred_file, header=False)

    qrels_english = DATA_PATH + english_data + "/gold.tsv"
    qrels_german = DATA_PATH + german_data + "/gold.tsv"

    gold_file = DATA_PATH + german_data + '_' + english_data + 'qrels.tsv'

    append_qrels_documents(qrels_english, qrels_german, gold_file, header=False)

    subprocess.call(["python", "evaluation/scorer/main.py", 'unknown', '-pred_file', pred_file, '-gold_file', gold_file])


if __name__ == "__main__":
    run()