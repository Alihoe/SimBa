import subprocess


def run():



    data_name_queries = "gesis_test_sup"
    data_name_cache = "gesis_test_sup"+"_more_text_pp_stop_words"
    data_name_targets = "gesis_unsup_more_text_pp_stop_words"
    data_name = "gesis_test_sup/gesis_test_sup"+"_more_text_pp_stop_words_st5"

    # data_name_cache = "gesis_test_sup"+"_text_pp_stop_words"
    # data_name_targets = "gesis_unsup_text_pp_stop_words"
    # data_name = "gesis_test_sup/gesis_test_sup"+"_text_pp_stop_words_st5"

    # data_name_cache = "gesis_test_sup"+"_labels_title_pp_stop_words"
    # data_name_targets = "gesis_unsup_labels_title_pp_stop_words"
    # data_name = "gesis_test_sup/gesis_test_sup"+"_labels_title_pp_stop_words_st5"

    # data_name_cache = "gesis_test_sup"+"_labels_pp_stop_words"
    # data_name_targets = "gesis_unsup_labels_pp_stop_words"
    # data_name = "gesis_test_sup/gesis_test_sup"+"_labels_pp_stop_words"



    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/"+data_name_queries+"/queries.tsv",
    #                  "../../data/"+data_name_targets+"/corpus",
    #                  data_name_cache,
    #                  data_name,
    #                  "braycurtis",
    #                  "100",
    #                  "--gesis_unsup",
    #                  "--ranking_only",
    #                  "--union",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base"])#, "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large",#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
    #                  #'-referential_similarity_measures', "spacy_ne_similarity"
    #                  #'-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  #'-string_similarity_measures', "sequence_matching", "levenshtein",
    #                  #'-discrete_similarity_measures', "spacy_no_nr_ne_similarity"
    #                  #])

    data_names = ["gesis_test_sup_labels", "gesis_test_sup_labels_pp_stop_words",
                  "gesis_test_sup_labels_title_st5", "gesis_test_sup_labels_title_pp_stop_words_st5",
                  "gesis_test_sup_text_st5", "gesis_test_sup_text_pp_stop_words_st5",
                  "gesis_test_sup_more_text_st5", "gesis_test_sup_more_text_pp_stop_words_st5",]

    for data_name in data_names:

        print("Evaluating "+data_name)

        subprocess.call(["python", "../../evaluation/scorer/main.py",
                         "../../data/"+data_name_queries+"/gold.tsv",
                         "../../data/"+data_name_queries+"/"+data_name+"/pred_qrels.tsv"])

if __name__ == "__main__":
    run()