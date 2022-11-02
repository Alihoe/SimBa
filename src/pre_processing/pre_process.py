from src.pre_processing.pre_process_tweets import remove_urls, replace_emojis

twitter_data = ["clef_2020_checkthat_2_english", "clef_2021_checkthat_2a_english", "clef_2021_checkthat_2b_english", "clef_2022_checkthat_2a_english", "clef_2022_checkthat_2b_english"]


def pre_process_twitter_data(queries):
    for key,value in queries.items():
        queries[key] = replace_emojis(remove_urls(value))
    return queries


def pre_process(queries, targets, data):
    if data in twitter_data:
        pre_process_twitter_data(queries)
    return queries, targets