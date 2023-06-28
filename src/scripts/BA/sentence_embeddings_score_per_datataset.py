import os
import subprocess
from os.path import realpath, dirname
import pandas as pd
import numpy as np
from evaluation.utils import get_ndcg_10


def run():

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parent_parent_dir_of_file = dirname(parent_dir_of_file)
    parents_parent_parent_dir_of_file = dirname(parent_parent_dir_of_file)
    repo_path = parents_parent_parent_dir_of_file
    data_path = repo_path + "/data/"

    similarity_features = ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "all-distilroberta-v1",
                           "sentence-transformers/sentence-t5-base", "princeton-nlp/unsup-simcse-roberta-base",
                           "princeton-nlp/sup-simcse-roberta-large", "https://tfhub.dev/google/universal-sentence-encoder/4",
                           "infersent"]

    data_names = ["nf", "scifact", "arguana",
                  "clef_2020_checkthat_2_english", "clef_2021_checkthat_2a_english",
                  "clef_2022_checkthat_2a_english", "cqa_dupstack_mathematica", "cqa_dupstack_webmasters",
                  "clef_2021_checkthat_2b_english", "clef_2022_checkthat_2b_english",
                  "cqa_dupstack_android", "scidocs", "cqa_dupstack_wordpress", "cqa_dupstack_programmers",
                  "cqa_dupstack_gis", "cqa_dupstack_physics",
                  "cqa_dupstack_english", "cqa_dupstack_stats", "cqa_dupstack_gaming",
                  "cqa_dupstack_unix", "fiqa", "cqa_dupstack_tex"]

    datasets_scores_df = pd.DataFrame(columns=['Dataset'] + similarity_features)
    datasets_scores_df['Dataset'] = data_names

    for data_name in data_names:

        print(data_name)

        dataset_path = repo_path + "/data/" + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_gold = dataset_path + "/gold.tsv"

        for own_idx, similarity_feature in enumerate(similarity_features):
            print(similarity_feature)

            if "/" or ":" or "." in str(similarity_feature):
                similarity_feature_path_name = str(similarity_feature).replace("/", "_").replace(":", "_").replace(".", "_")
            else:
                similarity_feature_path_name = similarity_feature

            data_name_pred = dataset_path + "/" + similarity_feature_path_name + "/pred_qrels.tsv"

            if not os.path.isfile(data_name_pred):
                subprocess.call(["python",
                                 repo_path + "/src/re_ranking/re_ranking.py",
                                 data_name_queries,
                                 data_name_targets,
                                 data_name,
                                 data_name + "/" + similarity_feature_path_name,
                                 "braycurtis",
                                 "10",
                                 '--ranking_only',
                                 '-sentence_embedding_models', similarity_feature])

            data_name_results = dataset_path + "/" + similarity_feature_path_name + "/results.tsv"

            if not os.path.isfile(data_name_results):

                print("Evaluation Scores for dataset " + data_name + "/" + similarity_feature)
                subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                                 data_name + "/" + similarity_feature_path_name,
                                 data_name_gold,
                                 data_name_pred])

            ndgc_10 = get_ndcg_10(repo_path + "/data/"+ data_name + "/" + similarity_feature_path_name)
            datasets_scores_df.loc[datasets_scores_df['Dataset'] == data_name, similarity_feature] = float(ndgc_10)

    print(datasets_scores_df)

    with open("BA_output/sentence_em_scores_per_dataset.txt", 'w') as f:
        print(datasets_scores_df.style.format_index(axis=1, formatter="${}$".format).hide(
            axis=0).format(precision=3).to_latex(column_format="|c|c|c|c|c|c|c|c|", position="h",
                                                 label="table:sentence_embedding_scores_per_dataset",
                                                 caption="Score of Different Sentence Embedding Models per Dataset.",
                                                 multirow_align="t", multicol_align="r"), file=f)


if __name__ == "__main__":
    run()