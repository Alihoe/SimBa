from src.pre_processing.pre_process_tweets import remove_urls, replace_emojis

twitter_data = ["clef_2020_checkthat_2_english", "clef_2021_checkthat_2a_english", "clef_2021_checkthat_2b_english", "clef_2022_checkthat_2a_english", "clef_2022_checkthat_2b_english"]
variable_data = ["sv_ident_en_train_and_val", "sv_ident_en_trial_de", "sv_ident_en_trial_de_test_and_train", "sv_ident_trial_de_train", "sv_ident_trial_en", "sv_ident_trial_en_test_and_train", "sv_ident_trial_en_train"]


def pre_process_variable_data(targets):
    return targets


def pre_process_twitter_data(queries):
    for key,value in queries.items():
        queries[key] = replace_emojis(remove_urls(value))
    return queries


def pre_process(queries, targets, data):
    if data in twitter_data:
        queries = pre_process_twitter_data(queries)
    if data in variable_data:
        targets = pre_process_variable_data(targets)
    return queries, targets