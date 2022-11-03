import argparse
import subprocess


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2022_checkthat_2a_english", help="Pass name of dataset stored in the data folder.")
    args = parser.parse_args()

    # subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval.py", args.data, "--pre_processing", "braycurtis", "--union_of_top_k_per_feature", "spearmanr", "50"])
    # subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking.py", args.data, "--pre_processing", "braycurtis", "spearmanr", "5"])
    subprocess.call(["python", "evaluation/scorer/main.py", args.data, "--pre_processing"])


if __name__ == "__main__":
    run()