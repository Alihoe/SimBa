import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def run():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('data', type=str, default="clef_2020_checkthat_2_english", help="Pass name of dataset stored in the data folder.")
    # args = parser.parse_args()

    data = "clef_2020_checkthat_2_english"
    Path("data/corpus_chunks/"+data).mkdir(parents=True, exist_ok=True)

    sizes = [1000, 5000, 10000]

    for corpus_size in sizes:
        # get the verified claims embeddings
        shutil.rmtree("data/corpus_chunks/" + data, ignore_errors=False, onerror=None)
        subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval_corpus_chunks.py", data, str(corpus_size),
             "braycurtis", "--union_of_top_k_per_feature", "spearman", "50", "-sentence_embedding_models", "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "https://tfhub.dev/google/universal-sentence-encoder/4"])
        # for fname in os.listdir("data/corpus_chunks/"+ data):
        #     if fname.startswith("embedded_queries"):
        #         os.remove(os.path.join("data/corpus_chunks/"+ data, fname))

        # Start measuring time here
        print("Measuring time for corpus size of " + str(corpus_size))
        start_time = time.time() # Measure time for retrieval and re-ranking
        subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval_corpus_chunks.py", data, str(corpus_size), "braycurtis", "--union_of_top_k_per_feature", "spearman", "50"])
        subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking_corpus_chunks.py", data, "braycurtis", "spearman", "50"])
        print("--- %s seconds ---" % (time.time() - start_time))
        # Check if output file was created correctly by evaluating with standard evaluation script
        # Don't measure exceution time for that
        subprocess.call(["python", "evaluation/scorer/main.py", data, "--use_corpus_chunk_data"])
        # Delete produced files
        #shutil.rmtree("data/corpus_chunks/"+ data, ignore_errors=False, onerror=None)



if __name__ == "__main__":
    run()