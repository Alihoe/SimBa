import os
import subprocess
from os.path import realpath, dirname

import pandas as pd

from evaluation.utils import get_ndcg_10


def run():

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parents_parent_dir_of_file = dirname(parent_dir_of_file)#
    parents_parents_parent_dir_of_file = dirname(parents_parent_dir_of_file)
    repo_path = parents_parents_parent_dir_of_file
    data_path = repo_path + "/data/"

    columns = ['Dataset', 'Number of Queries', 'Number of Targets', 'Average Query Length', 'Average Target Length', 'Score']

    comparison_df = pd.DataFrame(columns=columns)

    data_names = [
        "nf", "scifact", "arguana", "clef_2020_checkthat_2_english", "clef_2021_checkthat_2a_english",
                  "clef_2022_checkthat_2a_english", "cqa_dupstack_mathematica", "cqa_dupstack_webmasters",
                  "clef_2021_checkthat_2b_english", "clef_2022_checkthat_2b_english",
               "cqa_dupstack_android", "scidocs", "cqa_dupstack_wordpress", "cqa_dupstack_programmers",
                 "cqa_dupstack_gis", "cqa_dupstack_physics",
                  "cqa_dupstack_english", "cqa_dupstack_stats", "cqa_dupstack_gaming",
                 "cqa_dupstack_unix", "fiqa", "cqa_dupstack_tex"]


    for data_name in data_names:

        print(data_name)

        dataset_path = data_path + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_pred = dataset_path + "/pred_qrels.tsv"
        data_name_gold = dataset_path + "/gold.tsv"

        subprocess.call(["python", repo_path + "/evaluation/evaluate_datasets.py",
                         data_name,
                         data_name_queries,
                         data_name_targets
                         ])

        dataset_features = pd.read_csv(data_path + data_name + "/dataset_analysis.tsv", sep='\t')

        subprocess.call(["python",
                         repo_path + "/src/candidate_retrieval/retrieval.py",
                         data_name_queries,
                         data_name_targets,
                         data_name,
                         data_name,
                         "braycurtis",
                         "50",
                         '-sentence_embedding_models', "all-mpnet-base-v2"
                         ])

        subprocess.call(["python",
                         repo_path + "/src/re_ranking/re_ranking.py",
                         data_name_queries,
                         data_name_targets,
                         data_name,
                         data_name,
                         "braycurtis",
                         "10",
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base",
                         "princeton-nlp/unsup-simcse-roberta-base",
                         '-lexical_similarity_measures', "similar_words_ratio"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])
        dataset_features['Score'] = get_ndcg_10(repo_path + "/data/" + data_name)

        comparison_df = pd.concat([comparison_df, dataset_features])

    comparison_df = comparison_df.sort_values('Number of Targets').rename(columns={'Dataset': 'dataset', 'Number of Queries': '\# queries', 'Number of Targets': '\# targets', 'Average Query Length': 'avg. query length', 'Average Target Length': 'avg. target length', 'Score': 'score'})
    comparison_df.to_csv("BA_output/dataset_comparison.tsv", index=False, header=True, sep='\t')
    print(comparison_df.style.format_index(axis=1, formatter="${}$".format).hide(
    axis=0).format(precision=3).to_latex(column_format="c|c|c|c|c|c", position="h",
                                                    label="table:comp_datasets",
                                                    caption="Comparison of Datasets",
                                                    multirow_align="t", multicol_align="r"))
    with open("BA_output/dataset_comparison.txt", 'w') as f:
        print(comparison_df.style.format_index(axis=1, formatter="${}$".format).hide(
    axis=0).format(precision=3).to_latex(column_format="c|c|c|c|c|c", position="h",
                                                    label="table:comp_datasets",
                                                    caption="Comparison of Datasets",
                                                    multirow_align="t", multicol_align="r"), file=f)


if __name__ == "__main__":
    run()