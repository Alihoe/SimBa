import argparse
import subprocess
import random


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2020_checkthat_2a_english", help="Pass name of dataset stored in the data folder.")
    args = parser.parse_args()

    corpus_size_1 = 1000
    corpus_size_2 = 5000
    corpus_size_3 = 10000

    random.seed(2)

    corpus_path = "data/" + args.data + "/corpus"
    targets = get_targets(corpus_path)

    for i in range(4):
        some_dict.pop(random.choice(some_dict.keys()))


    subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval.py", args.data, "braycurtis", "--union_of_top_k_per_feature", "spearman", "100"])
    subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking.py", args.data, "braycurtis", "spearman", "50"])


    subprocess.call(["python", "evaluation/scorer/main.py", args.data])

    # subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval.py", args.data, "cosine", "--union_of_top_k_per_feature", "spearman", "50"])
    # subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking.py", args.data, "cosine", "spearman", "5"])
    # subprocess.call(["python", "evaluation/scorer/main.py", args.data])


if __name__ == "__main__":
    run()