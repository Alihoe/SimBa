import argparse
import shutil
import subprocess
import time


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2020_checkthat_2a_english", help="Pass name of dataset stored in the data folder.")
    args = parser.parse_args()

    sizes = [1000, 5000, 10000]

    for corpus_size in sizes:
        print("Measuring time for corpus size of "+str(corpus_size))
        start_time = time.time() # Measure time for retrieval and re-ranking
        subprocess.call(["python", "src/candidate_retrieval/semantic_retrieval_corpus_chunks.py", args.data, str(corpus_size), "braycurtis", "--union_of_top_k_per_feature", "spearman", "100"])
        subprocess.call(["python", "src/re_ranking/multi_feature_re_ranking_corpus_chunks.py", args.data, "braycurtis", "spearman", "50"])
        print("--- %s seconds ---" % (time.time() - start_time))
        # Check if output file was created correctly by evaluating with standard evaluation script
        # Don't measure exceution time for that
        subprocess.call(["python", "evaluation/scorer/main.py", args.data, "--use_corpus_chunk_data"])
        # Delete produced files
        shutil.rmtree("data/corpus_cunks/"+args.data, ignore_errors=False, onerror=None)


if __name__ == "__main__":
    run()