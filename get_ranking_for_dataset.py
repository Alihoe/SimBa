import argparse
import subprocess


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2022_checkthat_2a_english", help="Pass name of dataset stored in the data folder.")
    args = parser.parse_args()

    subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval.py", args.data, "braycurtis", "spearman", "50"])
    subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking.py", args.data, "braycurtis", "spearman", "50"])
    subprocess.call(["python", "evaluation/scorer/main.py", args.data])


if __name__ == "__main__":
    run()