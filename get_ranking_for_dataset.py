import argparse
import subprocess


def run():

    # id	study_title	variable_label	variable_name	answer_categories	topic	topic_en	question_text	question_text_en	sub_question	item_categories

    # variable_label topic_en question_text question_text_en
    #fields = ["-fields=variable_label", "topic_en", "question_text", "question_text_en"]


    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2022_checkthat_2a_english", help="Pass name of dataset stored in the data folder.")
    args = parser.parse_args()

    subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval.py", args.data, "braycurtis", "--union_of_top_k_per_feature", "spearmanr", "50", '-fields', 'variable_label', 'topic_en', 'question_text', 'question_text_en'])
    subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking.py", args.data, "braycurtis", "spearmanr", "10", '-fields', 'variable_label', 'topic_en', 'question_text', 'question_text_en'])
    subprocess.call(["python", "evaluation/scorer/main.py", args.data, '-fields', 'variable_label', 'topic_en', 'question_text', 'question_text_en'])


if __name__ == "__main__":
    run()