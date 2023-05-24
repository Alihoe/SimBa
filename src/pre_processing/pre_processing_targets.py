import argparse
from pathlib import Path
import pandas as pd
import lxml.html
from nltk import word_tokenize
from collections import Counter
from src.pre_processing import DATA_PATH
from src.utils import get_certain_target_fields


def filter_out_html(input):
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


def collect_tokens(input, tokens):
    try:
        tokens.extend(word_tokenize(str(input)))
    except:
        print("input")
        print(input)
        print("tokens")
        print(tokens)


def count_token_occurences(tokens):
    c = Counter(tokens)
    return c


def filter_out_k_most_common_tokens(token_dic, k):
    dict = {k: v for k, v in sorted(token_dic.items(), key=lambda item: item[1], reverse=True)}
    return list(dict.keys())[:k]


def filter_out_stop_words(input, stop_words):
    word_list = [w for w in word_tokenize(str(input)) if not w in stop_words]
    return ' '.join(word for word in word_list)



def run():
    """
    input:
    targets
    output:
    pre-processed targets

    Keep document structure intact
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', type=str, help='Input targets path as tsv file.')
    parser.add_argument('data', type=str, help='Name under which the documents should be stored.')
    parser.add_argument('-fields', type=str, nargs='+', default="all", help='Fields to keep in target file.')
    parser.add_argument('--html_filtering', action="store_true",
                        help='If selected all html elements are filtered out.')
    parser.add_argument('--stop_words', action="store_true",
                        help='Collect and filter out document dependent stop words.')
    args = parser.parse_args()

    output_dir = DATA_PATH + args.data +'/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    targets_df = pd.read_csv(args.targets, sep='\t', dtype=str)

    if args.fields != 'all' and args.fields != ['analysis']:
        fields = ['id']
        fields.extend(args.fields)
        output_path = output_dir + 'corpus'
        targets_df = get_certain_target_fields(args.targets, fields)
        targets_df.to_csv(output_path, index=False, header=True, sep='\t')

    if args.fields == ['analysis']:
        targets_df = pd.read_csv(args.targets, sep='\t', dtype=str)
        wanted_fields = []
        for field in targets_df.columns:
            example_content = targets_df.iloc[1][field]
            print(example_content)
            if example_content:
                if str(example_content) not in ["[]", "nan", "['variable']", "false"]:
                    if len(example_content) > 1:
                        if str(example_content)[0] != "<":
                            if str(field) not in ["study_citation_html_en",
                                                  "study_citation_html",
                                                  "question_type1",
                                                  "selection_method_en",
                                                  "selection_method_ddi_en",
                                                  "related_research_data",
                                                  "link_count",
                                                  "variable_name_sorting",
                                                  "type",
                                                  "countries_iso",
                                                  "study_group_en",
                                                  "time_method_en",
                                                  "group_link",
                                                  "group_link_en"
                                                  "study_citation_en",
                                                  "study_lang_en",
                                                  "selection_method_ddi",
                                                  "study_group",
                                                  "analysis_unit_en",
                                                  "group_image_file",
                                                  "variable_order",
                                                  "additional_keywords",
                                                  "data_source",
                                                  "index_source",
                                                  "study_citation",
                                                  "time_method",
                                                  "selection_method",
                                                  "group_description"]:
                                if field != "id.1":
                                    # if len(example_content) < 6
                                    wanted_fields.append(field)
        print(wanted_fields)
        output_path = output_dir + 'analysis_pp_targets.tsv'
        targets_df = get_certain_target_fields(args.targets, wanted_fields)
        targets_df.to_csv(output_path, index=False, header=True, sep='\t')

    if args.html_filtering:
        columns = targets_df.columns
        text_columns = columns[1:]
        for column in text_columns:
            targets_df[column] = targets_df[column].apply(filter_out_html)
        print(targets_df)
        output_path = output_dir + 'corpus'
        targets_df.to_csv(output_path, index=False, header=True, sep='\t')

    if args.html_filtering:
        columns = targets_df.columns
        text_columns = columns[1:]
        for column in text_columns:
            targets_df[column] = targets_df[column].apply(filter_out_html)
        output_path = output_dir + 'corpus'
        targets_df.to_csv(output_path, index=False, header=True, sep='\t')

    if args.stop_words:
        tokens = []
        columns = targets_df.columns
        text_columns = columns[1:]
        for column in text_columns:
            targets_df[column].apply(collect_tokens, args=(tokens,))
        token_dict = count_token_occurences(tokens)
        stop_words = filter_out_k_most_common_tokens(token_dict, 500)
        print(stop_words)
        print(len(stop_words))
        for column in text_columns:
            targets_df[column] = targets_df[column].apply(filter_out_stop_words, args=(stop_words,))
        print(targets_df)
        output_path = output_dir + 'corpus'
        targets_df.to_csv(output_path, index=False, header=True, sep='\t')




if __name__ == "__main__":
    run()


