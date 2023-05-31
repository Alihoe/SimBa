import pandas as pd

from src.get_data import DATA_PATH


def targets_to_queries(input_path, output_path):
    df = pd.read_csv(input_path, sep='\t', names=['id', 'variable_label', 'variable_label_en', 'title', 'title_en', 'question_text', 'question_text_en', 'answer_categories', 'answer_categories_en'], dtype=str)
    df['query'] = df[['variable_label', 'variable_label_en', 'title', 'title_en', 'question_text', 'question_text_en', 'answer_categories', 'answer_categories_en']].apply(lambda x: ''.join(x.astype(str)), axis=1)
    df = df.iloc[1:]
    # column_length = len(df.columns)
    # column_values = list(map(' '.join, df.iloc[:, 1:column_length].astype(str).values.tolist()))
    # queries = dict(zip(df.iloc[:, 0], column_values))
    # print(queries)
    # query_df = pd.DataFrame.from_dict(queries)
    df.to_csv(output_path, columns=['id', 'query'], sep='\t', header=False, index=False)


input_path = DATA_PATH + "gesis_unsup_more_text_pp" + "/corpus"
output_path = DATA_PATH + "gesis_unsup_more_text_pp" + "/queries.tsv"
targets_to_queries(input_path, output_path)
