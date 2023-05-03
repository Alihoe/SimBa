import os
import subprocess
from os.path import realpath, dirname

import pandas as pd

from evaluation.utils import get_ndcg_10


def run():

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parents_parent_dir_of_file = dirname(parent_dir_of_file)
    repo_path = parents_parent_dir_of_file

    similarity_features = [('-sentence_embedding_models', "all-mpnet-base-v2"),
                           ('-sentence_embedding_models', "sentence-transformers/sentence-t5-base"),
                           ('-sentence_embedding_models', "princeton-nlp/unsup-simcse-roberta-base"),
                           ('-referential_similarity_measures', "synonym_similarity"),
                           ('-referential_similarity_measures', "ne_similarity"),
                           ('-lexical_similarity_measures', "similar_words_ratio"),
                           ('-string_similarity_measures', "sequence_matching"),
                           ('-string_similarity_measures', "levenshtein"),
                           #('-string_similarity_measures', "jaccard_similarity"
                            ]

    columns = ['Dataset', 'Number of Queries', 'Number of Targets', 'Average Query Length', 'Average Target Length', 'Score']
    for similarity_feature in similarity_features:
        columns.append(similarity_feature[1])

    comparison_df = pd.DataFrame(columns=columns)

    data_names = [
        "nf", "scifact", "arguana", "clef_2020_checkthat_2_english", "clef_2021_checkthat_2a_english",
                  "clef_2022_checkthat_2a_english", "cqa_dupstack_mathematica", "cqa_dupstack_webmasters",
                  "clef_2021_checkthat_2b_english", "clef_2022_checkthat_2b_english",
               "cqa_dupstack_android", "scidocs", "cqa_dupstack_wordpress", "cqa_dupstack_programmers",
                 "cqa_dupstack_gis", "cqa_dupstack_physics",
                  "cqa_dupstack_english", "cqa_dupstack_stats", "cqa_dupstack_gaming",
                  "cqa_dupstack_unix", "fiqa", "cqa_dupstack_tex", "trec_covid",  "touche", "quora"]
                  #, "nq", "dbpedia", "hotpot_qa",  "fever", "climate-fever", "ms_marco"]

    for data_name in data_names:

        print(data_name)

        dataset_path = repo_path + "/data/" + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_pred = dataset_path + "/pred_qrels.tsv"
        data_name_gold = dataset_path + "/gold.tsv"

        subprocess.call(["python", repo_path + "/evaluation/evaluate_datasets.py",
                         data_name,
                         data_name_queries,
                         data_name_targets
                         ])

        dataset_features = pd.read_csv(repo_path + "/data/" + data_name + "/dataset_analysis.tsv", sep='\t')

        subprocess.call(["python",
                         repo_path + "/src/candidate_retrieval/retrieval.py",
                         data_name_queries,
                         data_name_targets,
                         data_name,
                         data_name,
                         "braycurtis",
                         "50",
                         "--correlation_analysis",
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base",
                         "princeton-nlp/unsup-simcse-roberta-base",
                         '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
                         '-lexical_similarity_measures', "similar_words_ratio",
                         '-string_similarity_measures', "sequence_matching", "levenshtein"])

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

        for similarity_feature in similarity_features:

            similarity_feature_category = similarity_feature[0]
            similarity_feature_name = similarity_feature[1]

            data_name_pred = dataset_path + "/" + similarity_feature_name + "/pred_qrels.tsv"

            subprocess.call(["python",
                             repo_path + "/src/re_ranking/re_ranking.py",
                             data_name_queries,
                             data_name_targets,
                             data_name,
                             data_name + "/" + similarity_feature_name,
                             "braycurtis",
                             "10",
                             '--ranking_only',
                             similarity_feature_category, similarity_feature_name])

            print("Evaluation Scores for dataset " + data_name + "/" + similarity_feature_name)
            subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                             data_name + "/" + similarity_feature_name,
                             data_name_gold,
                             data_name_pred])

            dataset_features[similarity_feature_name] = get_ndcg_10(repo_path + "/data/"+ data_name + "/" + similarity_feature_name)

        comparison_df = pd.concat([comparison_df, dataset_features])
        print(comparison_df)

        comparison_df = comparison_df.sort_values('Number of Targets')
        comparison_df.to_csv("comparison.tsv", index=False, header=True, sep='\t')


if __name__ == "__main__":
    run()