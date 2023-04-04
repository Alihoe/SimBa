import os
import subprocess
from os.path import realpath, dirname


def run():

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parents_parent_dir_of_file = dirname(parent_dir_of_file)

    data_names = ["nf", "trec_covid", "nq", "hotpot_qa", "fiqa","arguana", "touche",
                  "quora", "dbpedia", "scidocs", "fever", "climate-fever", "scifact",
                  "cqa_dupstack_android", "cqa_dupstack_english", "cqa_dupstack_gaming", "cqa_dupstack_gis",
                  "cqa_dupstack_mathematica", "cqa_dupstack_physics", "cqa_dupstack_programmers",
                  "cqa_dupstack_stats", "cqa_dupstack_tex", "cqa_dupstack_unix", "cqa_dupstack_webmasters",
                  "cqa_dupstack_wordpress", "ms_marco"]

    for data_name in data_names:

        print(data_name)

        subprocess.call(["python",
                         repo_path + "/src/candidate_retrieval/retrieval.py",
                         data_name_queries,
                         data_name_targets,
                         data_name,
                         data_name,
                         "braycurtis",
                         "50",
                         '-sentence_embedding_models', "all-mpnet-base-v2",
                         '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
                         '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
                         '-string_similarity_measures', "sequence_matching", "levenshtein", "jaccard"])

        similarity_features = [str('-sentence_embedding_models', "all-mpnet-base-v2"),
                               str('-referential_similarity_measures', "synonym_similarity"), "ne_similarity",
                               str('-referential_similarity_measures', "ne_similarity"),
                               str('-lexical_similarity_measures', "similar_words_ratio"),
                               str('-string_similarity_measures', "sequence_matching"),
                               str('-string_similarity_measures', "levenshtein"),
                               str('-string_similarity_measures', "jaccard")]

        for idx, similarity_feature in enumerate(similarity_features):

            repo_path = parents_parent_dir_of_file
            dataset_path = repo_path + "/data/" + data_name
            data_name_queries = dataset_path + "/queries.tsv"
            data_name_targets = dataset_path + "/corpus"
            data_name_gold = dataset_path + "/gold.tsv"
            data_name_pred = dataset_path + "/pred_qrels_" + str(idx)

            subprocess.call(["python",
                             repo_path + "/src/re_ranking/re_ranking.py",
                             data_name_queries,
                             data_name_targets,
                             data_name,
                             data_name,
                             "braycurtis",
                             "5",
                             '--ranking_only',
                             similarity_feature])

        print("Evaluation Scores for dataset " + data_name)
        subprocess.call(["python", repo_path + "/evaluation/scorer/scorer_main.py",
                         data_name_gold,
                         data_name_pred])

        subprocess.call(["python", repo_path + "/evaluation/scorer/ndcg_evaluator.py",
                         data_name_gold,
                         data_name_pred])



if __name__ == "__main__":
    run()