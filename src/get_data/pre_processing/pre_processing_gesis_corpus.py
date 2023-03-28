import json
import os
import pickle
from pathlib import Path

import pandas as pd

from src.get_data import DATA_PATH
from src.learning.variable_detection import load_classifier_and_predict_variables
from src.utils import load_pickled_object


def get_variable_dataset(corpus_id):
    example_file_path = DATA_PATH + "gesis_unsup/vocab/vocab/"+str(corpus_id)+".pkl"
    return pickle.load(open(example_file_path, 'rb'))


def tokenize_document(full_text):
    text_chunks = []
    return text_chunks


def create_queries_and_targets_for_document(document_n, queries_pp=False):

    queries_df = pd.DataFrame(columns=['uuid', 'text'])
    with open(DATA_PATH + "gesis_unsup/vadis_pubs_corpus.json") as f:
        documents_dict = json.load(f)
    document = list(documents_dict.items())[document_n]
    document_id = document[0]
    document_content = document[1]

    if queries_pp:
        Path(DATA_PATH + document_id+ "_pp").mkdir(parents=True, exist_ok=True)
        queries_data_path = DATA_PATH + document_id + "_pp/queries.tsv"
        targets_data_path = DATA_PATH + document_id + "_pp/corpus"
    else:
        Path(DATA_PATH + document_id).mkdir(parents=True, exist_ok=True)
        queries_data_path = DATA_PATH + document_id + "/queries.tsv"
        targets_data_path = DATA_PATH + document_id + "/corpus"

    document_fulltext = document_content["fulltext"]
    with open(DATA_PATH + document_id+ "/full_text.txt", "w", encoding="utf-8") as text_file:
        text_file.write(document_fulltext)
    document_sentences = document_content["sentence_tokens"]
    queries_df['text'] = pd.Series(document_sentences)
    queries_df['uuid'] = queries_df.index

    queries_df.to_csv(queries_data_path, sep='\t', header=False, index=False)

    if queries_pp:
        load_classifier_and_predict_variables(queries_data_path)

    related_datsets = document_content["related_research_datasets"]

    for ds in related_datsets:
        variables = get_variable_dataset(ds)
        all_variable_parts = []
        variable_fields_info = {}
        for variable in variables.items():
            variable_fields = variable[1]['_source']
            for field in variable_fields.keys():
                variable_fields_info[field] = 0
        for variable in variables.items():
            variable_fields = variable[1]['_source']
            for field in variable_fields.keys():
                variable_fields_info[field] += 1
            all_variable_parts.extend(list(variable_fields.keys()))
    print(variable_fields_info)
    all_variable_parts = list(set(all_variable_parts))
    columns = ['id'] + all_variable_parts
    corpus_df = pd.DataFrame(columns=columns)
    for ds in related_datsets:
        variables = get_variable_dataset(ds)
        for variable in variables.items():
            this_row = []
            id = variable[0]
            this_row = this_row + [id]
            variable_fields = variable[1]['_source']
            for column in columns[1:]:
                if column in variable_fields.keys():
                    this_row = this_row + [variable_fields[column]]
                else:
                    this_row = this_row + [" "]
            target_df = pd.DataFrame([this_row], columns=columns)
            corpus_df = pd.concat([corpus_df, target_df], names=columns)
    corpus_df.to_csv(targets_data_path, sep='\t', header=True, index=False)

#create_queries_and_targets_for_document(1)#, queries_pp=True)
#create_queries_and_targets_for_document(2, queries_pp=True)
#create_queries_and_targets_for_document(3, queries_pp=True)
#create_queries_and_targets_for_document(10)
#create_queries_and_targets_for_document(10, queries_pp=True)


def create_queries_for_document(document_n, queries_pp=False):

    queries_df = pd.DataFrame(columns=['uuid', 'text'])
    with open(DATA_PATH + "gesis_unsup/sentence_extractions/"+str(document_n)+".json", encoding='utf-8') as f:
        document = json.load(f)

    if queries_pp:
        Path(DATA_PATH + document_n+"/"+document_n+ "_pp").mkdir(parents=True, exist_ok=True)
        queries_data_path = DATA_PATH + document_n+"/"+document_n + "_pp/queries.tsv"
    else:
        Path(DATA_PATH + document_n).mkdir(parents=True, exist_ok=True)
        queries_data_path = DATA_PATH + document_n + "/queries.tsv"

    document_sentences = document["sentences"]
    queries_df['text'] = pd.Series(document_sentences)
    queries_df['uuid'] = queries_df.index

    queries_df.to_csv(queries_data_path, sep='\t', header=False, index=False)

    if queries_pp:
        load_classifier_and_predict_variables(queries_data_path)


create_queries_for_document("74901", queries_pp=True)
create_queries_for_document("79409", queries_pp=True)
create_queries_for_document("76005", queries_pp=True)
create_queries_for_document("79636", queries_pp=True)
create_queries_for_document("75199", queries_pp=True)
create_queries_for_document("77659", queries_pp=True)
create_queries_for_document("75302", queries_pp=True)
create_queries_for_document("76981", queries_pp=True)
create_queries_for_document("74465", queries_pp=True)
create_queries_for_document("72164", queries_pp=True)
create_queries_for_document("73518", queries_pp=True)
create_queries_for_document("44346", queries_pp=True)
create_queries_for_document("79197", queries_pp=True)
create_queries_for_document("75202", queries_pp=True)
create_queries_for_document("79639", queries_pp=True)
create_queries_for_document("33888", queries_pp=True)


def get_all_queries():

    all_docs = os.listdir(path=DATA_PATH + "gesis_unsup/sentence_extractions")
    all_queries_df = pd.DataFrame(columns=['uuid', 'text'])

    queries_data_path = DATA_PATH + "gesis_unsup/all_sentences.tsv"

    for doc in all_docs:
        queries_df = pd.DataFrame(columns=['uuid', 'text'])
        with open(DATA_PATH + "gesis_unsup/sentence_extractions/"+doc, encoding='utf-8') as f:
            document = json.load(f)

        document_sentences = document["sentences"]
        queries_df['text'] = pd.Series(document_sentences)
        queries_df['uuid'] = queries_df.index
        queries_df['uuid'] = queries_df['uuid'].astype(str)+str(doc)[:-6]

        all_queries_df = pd.concat([all_queries_df, queries_df])

    all_queries_df.to_csv(queries_data_path, sep='\t', header=False, index=False)

#get_all_queries()

def prepare_all_targets():
    all_docs = os.listdir(path=DATA_PATH + "gesis_unsup/vocab")

    for doc in all_docs:
        with open(DATA_PATH + "gesis_unsup/vocab/"+doc, "rb") as f:
            variables = pickle.load(f)
        all_variable_parts = []
        variable_fields_info = {}
        for variable in variables.items():
            variable_fields = variable[1]['_source']
            for field in variable_fields.keys():
                variable_fields_info[field] = 0
                variable_fields_info[field] += 1
            all_variable_parts.extend(list(variable_fields.keys()))

    all_variable_parts = list(set(all_variable_parts))
    columns = ['id'] + all_variable_parts
    corpus_df = pd.DataFrame(columns=columns)
    for doc in all_docs:
        with open(DATA_PATH + "gesis_unsup/vocab/"+doc, "rb") as f:
            variables = pickle.load(f)
        for variable in variables.items():
            this_row = []
            id = variable[0]
            this_row = this_row + [id]
            variable_fields = variable[1]['_source']
            for column in columns[1:]:
                if column in variable_fields.keys():
                    this_row = this_row + [variable_fields[column]]
                else:
                    this_row = this_row + [" "]
            target_df = pd.DataFrame([this_row], columns=columns)
            corpus_df = pd.concat([corpus_df, target_df], names=columns)
    targets_data_path = DATA_PATH + "gesis_unsup/corpus"
    corpus_df.to_csv(targets_data_path, sep='\t', header=True, index=False)

#prepare_all_targets()


def prepare_all_targets():
    all_docs = os.listdir(path=DATA_PATH + "gesis_unsup/vocab")
    all_variable_parts = []
    variable_fields_info = {}
    for doc in all_docs:
        with open(DATA_PATH + "gesis_unsup/vocab/"+doc, "rb") as f:
            variables = pickle.load(f)
        for variable in variables.items():
            variable_fields = variable[1]['_source']
            for field in variable_fields.keys():
                if field in variable_fields_info.keys():
                    variable_fields_info[field] += 1
                else:
                    variable_fields_info[field] = 0
            all_variable_parts.extend(list(variable_fields.keys()))

    all_variable_parts = list(set(all_variable_parts))
    print(all_variable_parts)
    columns = ['id'] + all_variable_parts
    print(columns)
    corpus_df = pd.DataFrame(columns=columns)
    for doc in all_docs:
        with open(DATA_PATH + "gesis_unsup/vocab/"+doc, "rb") as f:
            variables = pickle.load(f)
        for variable in variables.items():
            this_row = []
            id = variable[0]
            this_row.append(id)
            variable_fields = variable[1]['_source']
            for column in columns[1:]:
                if column in variable_fields.keys():
                    this_row.append(variable_fields[column])
                else:
                    this_row.append("")
            target_df = pd.DataFrame([this_row], columns=columns)
            corpus_df = pd.concat([corpus_df, target_df], names=columns)
    targets_data_path = DATA_PATH + "gesis_unsup/corpus_all"
    corpus_df.to_csv(targets_data_path, sep='\t', header=True, index=False)

#prepare_all_targets()





