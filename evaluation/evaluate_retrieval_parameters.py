import argparse
import subprocess

#["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"],

#"-sentence_embedding_models", "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2022_checkthat_2a_english", help="Pass name of dataset stored in the data folder.")
    args = parser.parse_args()

    subprocess.call(["python", "../src/candidate_retrieval/semantic_retrieval.py", args.data, "braycurtis", "--union_of_top_k_per_feature", "spearmanr", "20", "-sentence_embedding_models", "all-mpnet-base-v2"])
    subprocess.call(["python", "../evaluation/scorer/recall_evaluator.py", args.data])

    subprocess.call(["python", "../src/re_ranking/multi_feature_re_ranking.py", args.data, "braycurtis", "spearmanr", "50"])
    subprocess.call(["python", "../evaluation/scorer/main.py", args.data])
    subprocess.call(["python", "../evaluation/scorer/recall_evaluator.py", args.data, "--after_re_ranking"])


if __name__ == "__main__":
    run()