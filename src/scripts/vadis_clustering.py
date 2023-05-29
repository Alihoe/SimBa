import subprocess


def run():

    data_name_targets = "gesis_unsup_text"

    for k in range(5, 6):

        data_name = "clustering_" + str(k)

        # subprocess.call(["python",
        #                  "../../src/clustering/create_clusters.py",
        #                  data_name_targets,
        #                  data_name,
        #                  str(k),
        #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base"])#, "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large",#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
        #                  #'-referential_similarity_measures', "spacy_ne_similarity"
        #                  #'-lexical_similarity_measures', "similar_words_ratio"
        #                  #])


        data_name_queries = "11155/11155_pp"

        subprocess.call(["python",
                         "../../src/clustering/query_to_cluster.py",
                         data_name_queries,
                         data_name,
                         '-sentence_embedding_models', "sentence-transformers/sentence-t5-base"])#, "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large",#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
                         #'-referential_similarity_measures', "spacy_ne_similarity"
                         #'-lexical_similarity_measures', "similar_words_ratio"
                         #])

        data_name_cache = data_name_queries

        subprocess.call(["python",
                         "../../src/re_ranking/re_ranking.py",
                         "../../data/"+data_name_queries+"/queries.tsv",
                         "../../data/"+data_name_targets+"/corpus",
                         data_name_cache,
                         data_name,
                         "braycurtis",
                         "1",
                         "--gesis_unsup",
                         "--union",
                         '-sentence_embedding_models', "sentence-transformers/sentence-t5-base"])#, "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large",#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
                         #'-referential_similarity_measures', "spacy_ne_similarity"
                         #'-lexical_similarity_measures', "similar_words_ratio",
        # "similar_words_ratio_length",
                         #'-string_similarity_measures', "sequence_matching", "levenshtein",
                         #'-discrete_similarity_measures', "spacy_no_nr_ne_similarity"
                         #])


if __name__ == "__main__":
    run()