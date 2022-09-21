import subprocess

subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval.py", "clef_2022_checkthat_2a_english", "braycurtis", "spearman", "50"])
subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking.py", "clef_2022_checkthat_2a_english", "braycurtis", "spearman", "50"])
subprocess.call(["python", "evaluation/scorer/main.py", "../../data/clef_2022_checkthat_2a_english/pred_qrels.tsv", "../../data/clef_2022_checkthat_2a_english/gold.tsv"])

