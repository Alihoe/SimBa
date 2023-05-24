from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

import os
import pickle
import pandas as pd


additional_stop_words = ["nan", "gesis"]


def prepare_all_variables(vocab_path: str, targets_data_path: str, fields: list = ["variable_label", "variable_label_en", "title", "title_en", "question_text", "question_text_en", "answer_categories", "answer_categories_en"]):
    all_docs = os.listdir(path=vocab_path)
    columns = ['id'] + fields
    corpus_df = pd.DataFrame(columns=columns)
    for doc in all_docs:
        with open(vocab_path + "/" + doc, "rb") as f:
            variables = pickle.load(f)
        for variable in variables.items():
            this_row = []
            id = variable[0]
            this_row.append(id)
            variable_fields = variable[1]['_source']
            available_fields = variable_fields.keys()
            for column in columns[1:]:
                if column in available_fields:
                    this_row.append(variable_fields[column])
                else:
                    this_row.append("")
            target_df = pd.DataFrame([this_row], columns=columns)
            corpus_df = pd.concat([corpus_df, target_df], names=columns)
    corpus_df.to_csv(targets_data_path, sep='\t', header=True, index=False)


def filter_out_html(input: str) -> str:
    input_1 = str(input).replace("<th>", " ")
    input_2 = str(input_1).replace("<tr>", "")
    input_3 = str(input_2).replace("<td>", " ")
    input_4 = str(input_3).replace("</th>", " ")
    input_5 = str(input_4).replace("</tr>", "")
    input_6 = str(input_5).replace("</td>", " ")
    input_7 = str(input_6).replace("<br>", "")
    input_8 = str(input_7).replace("</br>", "")
    input_9 = str(input_8).replace("<br/>", "")
    input_10 = str(input_9).replace("<th/>", "")
    input_11 = str(input_10).replace("<td/>", "")
    input_12 = str(input_11).replace("<tr/>", "")
    input_13 = str(input_12).replace("<table class=", "")
    input_14 = str(input_13).replace("variables_answer_categories\">", "")
    input_15 = str(input_14).replace("\"", "")
    input_16 = str(input_15).replace("</table>", "")
    return input_16


def filter_out_k_most_common_tokens(token_dic: dict, k: int) -> list:
    dict = {k: v for k, v in sorted(token_dic.items(), key=lambda item: item[1], reverse=True)}
    return list(dict.keys())[:k]


def collect_tokens(input, tokens):
    tokens.extend(word_tokenize(str(input.lower())))


def find_typical_variable_words(targets_data_path: str, n: int = 500) -> list:
    targets_df = pd.read_csv(targets_data_path, sep='\t', dtype=str)
    tokens = []
    columns = targets_df.columns
    text_columns = columns[1:]
    for column in text_columns:
        targets_df[column] = targets_df[column].apply(filter_out_html)
    text_columns = columns[1:]
    for column in text_columns:
        targets_df[column].apply(collect_tokens, args=(tokens,))
    token_dict = Counter(tokens)
    typical_words = filter_out_k_most_common_tokens(token_dict, n)
    return typical_words


def filter_out_stop_words(input_words: list) -> list:
    relevant_stopwords = stopwords.words('english') + stopwords.words('german') + additional_stop_words
    return [word for word in input_words if word not in relevant_stopwords and word.isalpha() and not(word.isdigit())]


def store_typical_variable_words(vocab_path: str, n: int = 500, fields: list = ["variable_label", "variable_label_en", "title", "title_en", "question_text", "question_text_en", "answer_categories", "answer_categories_en"]):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'data')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    field_code = '_'.join([field_name[:1]+field_name[-1:] for field_name in fields])
    if len(field_code) > 100:
        field_code = ''.join([field_name[:1] for field_name in fields])
    output_path = final_directory + "/" + field_code + str(n) + "_words.txt"
    if os.path.exists(output_path):
        filtered_words = open(output_path, 'r').read().split('\n')
    else:
        targets_data_path = final_directory + "/" + field_code + "_corpus.tsv"
        if not os.path.exists(targets_data_path):
            prepare_all_variables(vocab_path=vocab_path, targets_data_path=targets_data_path, fields=fields)
        typical_words = find_typical_variable_words(targets_data_path=targets_data_path, n=n)
        filtered_words = filter_out_stop_words(typical_words)
        with open(output_path, 'w') as f:
            for item in filtered_words:
                f.write("%s\n" % item)
    return filtered_words


def contains_typical_words(sentence: str, token_list: list, k: int = 1) -> bool:
    """Detects sentences that have at least k tokens in common with a given list of tokens.

    Args:
      sentence: A document sentence as string.
      token_list: A list of tokens to compare sentence tokens to.
      k: A threshold of whether the input sentence has enough tokens in common with the token list.

    Returns:
      A Boolean of whether the sentence has at least k tokens in common with the given list of tokens.
    """
    sentence_tokens = []
    collect_tokens(sentence, sentence_tokens)
    if len(list(set(sentence_tokens).intersection(token_list))) >= k:
        return True
    else:
        return False


def find_sentences_with_typical_words(vocab_path: str, sentences: list, n : int = 1000, k : int = 2, fields: list = ["additional_keywords", "question_text", "methodology_collection_ddi_en", "group_description_en", "selection_method", "question_text_en", "variable_interview_instructions_en", "study_group", "answer_categories", "question_type2", "item_categories", "title_en", "title", "study_citation_en", "variable_label_en", "variable_name", "type", "item_categories_en", "selection_method_en", "codebook_table_prop", "study_id_title_en", "study_citation_html", "question_type1", "selection_method_ddi_en",  "question_name", "study_id", "study_title_en", "methodology_collection_dd", "time_method", "analysis_unit_en", "group_description", "notes_en", "notes",  "variable_interview_instructions", "methodology_collection_en", "sub_question", "study_title",  "analysis_unit", "study_citation", "study_group_en", "study_citation_html_en",  "study_id_title", "codebook_table_html"]):
    """Creates list of typical tokens in variables and filters out sentence that do not have at least k tokens in common with that list.
    Variables are taken from a directory of pkl files and get stored in a separate tsv document.
    The list of typical tokens gets stored as a txt file.

    Args:
      sentences: A list of document sentences.
      vocab_path: Path to directory where variables are stored.
      n: How many tokens of the original documents should be considered.
      k: A threshold of whether a sentence has enough tokens in common with the token list.
      fields: Variable fields to be considered for the creation of the token list.

    Returns:
      A Boolean of whether the sentence has at least k tokens in common with the given list of tokens.
    """
    typical_words = store_typical_variable_words(vocab_path=vocab_path, n=n, fields=fields)
    sentences_with_typical_words = [sentence for sentence in sentences if contains_typical_words(sentence, typical_words, k)]
    return sentences_with_typical_words
