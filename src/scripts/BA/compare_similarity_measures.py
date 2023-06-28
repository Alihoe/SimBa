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

    similarity_features = [('-sentence_embedding_models', "all-mpnet-base-v2"),
                           ('-sentence_embedding_models', "multi-qa-mpnet-base-dot-v1"),
                           ('-sentence_embedding_models', "all-distilroberta-v1"),
                           ('-sentence_embedding_models', "sentence-transformers/sentence-t5-base"),
                           ('-sentence_embedding_models', "princeton-nlp/unsup-simcse-roberta-base"),
                           ('-sentence_embedding_models', "princeton-nlp/sup-simcse-roberta-large"),
                           ('-sentence_embedding_models', "https://tfhub.dev/google/universal-sentence-encoder/4"),
                           ('-sentence_embedding_models', "infersent"),
                           ('-referential_similarity_measures', "synonym_similarity"),
                           ('-referential_similarity_measures', "ne_similarity"),
                           ('-lexical_similarity_measures', "similar_words_ratio"),
                           ('-string_similarity_measures', "sequence_matching"),
                           ('-string_similarity_measures', "levenshtein")]

    skip = 0

    for own_idx, similarity_feature in enumerate(similarity_features[skip:]):
        print(similarity_feature)
        own_idx = own_idx + skip

        similarity_feature_category = similarity_feature[0]
        similarity_feature_name = similarity_feature[1]
        cleaned_similarity_feature_name = str(similarity_feature_name).replace("/", "_").replace(":", "_").replace(".", "_")
        output_path = dir_of_file + "/BA_output/" + cleaned_similarity_feature_name+".txt"

        data_names = ["nf", "scifact", "arguana",
                      "clef_2020_checkthat_2_english", "clef_2021_checkthat_2a_english",
                      "clef_2022_checkthat_2a_english", "cqa_dupstack_mathematica", "cqa_dupstack_webmasters",
                      "clef_2021_checkthat_2b_english", "clef_2022_checkthat_2b_english",
                      "cqa_dupstack_android", "scidocs", "cqa_dupstack_wordpress", "cqa_dupstack_programmers",
                      "cqa_dupstack_gis", "cqa_dupstack_physics",
                      "cqa_dupstack_english", "cqa_dupstack_stats", "cqa_dupstack_gaming",
                      "cqa_dupstack_unix", "fiqa", "cqa_dupstack_tex"]

        performances_for_datasets = {}
        performances_two_similarity_scores_for_datasets = {}

        correlations = []

        for data_name in data_names:

            dataset_path = repo_path + "/data/" + data_name
            data_name_queries = dataset_path + "/queries.tsv"
            data_name_targets = dataset_path + "/corpus"
            data_name_gold = dataset_path + "/gold.tsv"

            # See how this score correlates with other similarity scores for this document

            correlation_file = dataset_path + "/" + data_name + "_correlation.tsv"

            if not os.path.isfile(correlation_file):

                subprocess.call(["python",
                                 repo_path + "/src/candidate_retrieval/retrieval.py",
                                 data_name_queries,
                                 data_name_targets,
                                 data_name,
                                 data_name,
                                 "braycurtis",
                                 "50",
                                 "--correlation_analysis",
                                 '-sentence_embedding_models',
                                 "all-mpnet-base-v2",
                                 "multi-qa-mpnet-base-dot-v1",
                                 "all-distilroberta-v1",
                                 "sentence-transformers/sentence-t5-base",
                                 "princeton-nlp/unsup-simcse-roberta-base",
                                 "princeton-nlp/sup-simcse-roberta-large",
                                 "https://tfhub.dev/google/universal-sentence-encoder/4",
                                 "infersent",
                                 '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
                                 '-lexical_similarity_measures', "similar_words_ratio",
                                 '-string_similarity_measures', "sequence_matching", "levenshtein"])

            this_dataset_correlations = pd.read_csv(dataset_path + "/" + data_name + "_correlation.tsv", sep='\t')
            this_sim_score_correlations = this_dataset_correlations[similarity_feature_name]
            correlations.append(this_sim_score_correlations)

            # Collect predictions for this dataset with specific similarity score

            if "/" or ":" or "." in str(similarity_feature_name):
                similarity_feature_path_name = str(similarity_feature_name).replace("/", "_").replace(":", "_").replace(".", "_")

            data_name_pred = dataset_path + "/" + similarity_feature_name + "/pred_qrels.tsv"

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
                                 similarity_feature_category, similarity_feature_name])

            data_name_results = dataset_path + "/" + similarity_feature_name + "/results.tsv"

            if not os.path.isfile(data_name_results):

                print("Evaluation Scores for dataset " + data_name + "/" + similarity_feature_name)
                subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                                 data_name + "/" + similarity_feature_path_name,
                                 data_name_gold,
                                 data_name_pred])

            ndgc_10 = get_ndcg_10(repo_path + "/data/" + data_name + "/" + similarity_feature_name)
            performances_for_datasets[data_name] = float(ndgc_10)

            # Look how similarity score performs with another similarity score for this dataset

            performances_two_similarity_scores_for_datasets[data_name] = {}
            all_other_similarity_features = []

            for second_similarity_feature in similarity_features:
                if second_similarity_feature != similarity_feature:
                    second_similarity_feature_category = second_similarity_feature[0]
                    second_similarity_feature_name = second_similarity_feature[1]

                    if "/" or ":" or "." in str(second_similarity_feature_name):
                        second_similarity_feature_path_name = str(second_similarity_feature_name).replace("/", "_").replace(":", "_")\
                            .replace(".", "_")

                    all_other_similarity_features.append(second_similarity_feature_name)

                    this_data_name = data_name + "/" + similarity_feature_name + "/" + second_similarity_feature_path_name

                    data_name_pred = data_path + this_data_name + "/pred_qrels.tsv"

                    if not os.path.isfile(data_name_pred):

                        if second_similarity_feature_category != similarity_feature_category:
                            subprocess.call(["python",
                                             repo_path + "/src/re_ranking/re_ranking.py",
                                             data_name_queries,
                                             data_name_targets,
                                             data_name,
                                             this_data_name,
                                             "braycurtis",
                                             "10",
                                             '--ranking_only',
                                             similarity_feature_category, similarity_feature_name,
                                             second_similarity_feature_category, second_similarity_feature_name])
                        else:
                            subprocess.call(["python",
                                             repo_path + "/src/re_ranking/re_ranking.py",
                                             data_name_queries,
                                             data_name_targets,
                                             data_name,
                                             this_data_name,
                                             "braycurtis",
                                             "10",
                                             '--ranking_only',
                                             similarity_feature_category, similarity_feature_name, second_similarity_feature_name])

                    data_name_results = data_path + this_data_name + "/results.tsv"

                    if not os.path.isfile(data_name_results):
                        print("Evaluation Scores for dataset " + this_data_name)
                        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                                         this_data_name,
                                         data_name_gold,
                                         data_name_pred])

                    ndcg_10 = get_ndcg_10(data_path + this_data_name)
                    performances_two_similarity_scores_for_datasets[data_name][second_similarity_feature_name] = float(ndcg_10)

        sorted_performances_for_datasets = dict(sorted(performances_for_datasets.items(), key=lambda x:x[1], reverse=True))

        mean_correlations = np.mean(correlations, axis=0).tolist()
        del mean_correlations[own_idx]

        means_of_perf_diff_second_sim_feature = []
        sim_features_perform_diff = {}

        for sim_feature in all_other_similarity_features:
            sim_features_perform_diff[sim_feature] = []

        for dataset, performances in performances_two_similarity_scores_for_datasets.items():
            for sim_feature in all_other_similarity_features:
                sim_features_perform_diff[sim_feature].append(performances[sim_feature] - sorted_performances_for_datasets[dataset])

        for sim_feature in all_other_similarity_features:
            means_of_perf_diff_second_sim_feature.append(np.mean(sim_features_perform_diff[sim_feature], axis=0))

        with open(output_path, 'w') as f:
            print(similarity_feature_name, file=f)
            print(sorted_performances_for_datasets, file=f)
            print(all_other_similarity_features, file=f)
            print(mean_correlations, file=f)
            print(performances_two_similarity_scores_for_datasets, file=f)
            print(means_of_perf_diff_second_sim_feature, file=f)

        dataset_df = pd.DataFrame(columns=['Dataset', 'Score'])
        dataset_df['Dataset'] = list(sorted_performances_for_datasets.keys())
        dataset_df['Score'] = list(sorted_performances_for_datasets.values())

        print(dataset_df.style.format_index(axis=1, formatter="${}$".format).hide(
            axis=0).format(precision=3).to_latex(column_format="c|c", position="h",
                                                 label="table:score_per_dataset_" + similarity_feature_name,
                                                 caption="Scores for Datasets produced by "+similarity_feature_name,
                                                 multirow_align="t", multicol_align="r"))
        with open("BA_output/sim_features_per_dataset_comparison_" + similarity_feature_path_name + ".txt", 'w') as f:
            print(dataset_df.style.format_index(axis=1, formatter="${}$".format).hide(
                axis=0).format(precision=3).to_latex(column_format="c|c", position="h",
                                                     label="table:score_per_dataset_" + similarity_feature_name,
                                                     caption="Scores for Datasets produced by " + similarity_feature_name,
                                                     multirow_align="t", multicol_align="r"), file=f)

        ensemble_df = pd.DataFrame(columns=['Similarity Feature', 'Correlation', 'Improvement'])
        ensemble_df['Similarity Feature'] = all_other_similarity_features
        ensemble_df['Correlation'] = mean_correlations
        ensemble_df['Improvement'] = means_of_perf_diff_second_sim_feature

        print(ensemble_df.style.format_index(axis=1, formatter="${}$".format).hide(
            axis=0).format(precision=3).to_latex(column_format="c|c|c", position="h",
                                                 label="table:ensemble_" + similarity_feature_name,
                                                 caption="Correlation of "+similarity_feature_name + " with other features.",
                                                 multirow_align="t", multicol_align="r"))
        with open("BA_output/sim_features_ensemble_" + similarity_feature_path_name + ".txt", 'w') as f:
            print(ensemble_df.style.format_index(axis=1, formatter="${}$".format).hide(
                axis=0).format(precision=3).to_latex(column_format="c|c|c", position="h",
                                                     label="table:ensemble_" + similarity_feature_name,
                                                     caption="Correlation of "+similarity_feature_name + " with other features.",
                                                     multirow_align="t", multicol_align="r"), file=f)


if __name__ == "__main__":
    run()