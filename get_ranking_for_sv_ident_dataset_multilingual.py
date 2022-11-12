import subprocess


def run():

    multilingual_data = 'sv_ident_train_and_val'

    subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval.py", multilingual_data, "cosine", "--union_of_top_k_per_feature", "spearmanr", "50", '-sentence_embedding_models', 'distiluse-base-multilingual-cased-v1'])
    subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking.py", multilingual_data, "cosine", "spearmanr", "10", '-sentence_embedding_models', 'distiluse-base-multilingual-cased-v1'])

    subprocess.call(["python", "evaluation/scorer/main.py", multilingual_data])


if __name__ == "__main__":
    run()