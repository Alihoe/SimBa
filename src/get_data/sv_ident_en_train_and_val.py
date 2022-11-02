import os
from os.path import join
from os import listdir, rmdir
from shutil import move
import gdown
import json

import shutil

from zipfile import ZipFile
from pathlib import Path

import pandas as pd
import requests


def run():

    # queries
    # qrels/gold
    # targets

    data_name = "sv_ident_en_train_and_val"

    Path("../../data/"+data_name).mkdir(parents=True, exist_ok=True)
    general_path = "../../data/"+data_name

    queries_path_train = general_path + "/queries_train.tsv"
    queries_path_val = general_path + "/queries_val.tsv"

    queries_path_en = general_path + "/queries.tsv"
    qrels_path_en = general_path + "/gold.tsv"

    queries_qrels_val = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/train/val.tsv")
    queries_qrels_train = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/train/train.tsv")

    with open(queries_path_val, 'wb') as f:
        f.write(queries_qrels_val.content)

    with open(queries_path_train, 'wb') as f:
        f.write(queries_qrels_train.content)

    df_en_val = pd.read_csv(queries_path_val, sep='\t')
    df_en_val = df_en_val[df_en_val.lang == "en"]
    df_en_train = pd.read_csv(queries_path_train, sep='\t')
    df_en_train = df_en_train[df_en_train.lang == "en"]

    df_en = pd.concat([df_en_val, df_en_train])
    df_en = df_en.loc[df_en['is_variable'] != 0]

    queries_df_en =  pd.DataFrame(columns=['uuid', 'text'])
    qrels_df_en = pd.DataFrame(columns=['uuid', '0', 'variable', '1'])

    research_data = []

    for index, row in df_en.iterrows():
        uuid = row['uuid']
        sentence = row['sentence']
        variables = row['variable'].split(";")
        if variables and variables != ["unk"]:
            current_research_data = row['research_data'].split(";")
            research_data.extend(current_research_data)
        for variable in variables:
            if variable != "unk":
                qrels_df = pd.DataFrame([[uuid, 0, variable, 1]], columns=['uuid', '0', 'variable', '1'])
                qrels_df_en = pd.concat([qrels_df_en, qrels_df])
                queries_df = pd.DataFrame([[uuid, sentence]], columns=['uuid', 'text'])
                queries_df_en = pd.concat([queries_df_en, queries_df])

    queries_df_en = queries_df_en.drop_duplicates()

    qrels_df_en.to_csv(qrels_path_en, sep= '\t', header = False, index=False)
    queries_df_en.to_csv(queries_path_en, sep= '\t', header = False, index=False)

    os.remove(general_path + "/queries_train.tsv")
    os.remove(general_path + "/queries_val.tsv")

    targets_path_en_dic = general_path + "/corpus_dic"
    targets_path_en_df = general_path + "/corpus"
    targets_en_url = "https://drive.google.com/uc?id=18slgACOcE8-_xIDX09GrdpFSqRRcBiON&export=download"
    gdown.download(targets_en_url, targets_path_en_dic, quiet=False)

    with open(targets_path_en_dic) as f:
        data = f.read()
    corpus_dict = json.loads(data)

    target_fields = []

    variable_chunks = corpus_dict.values()
    for variable_chunk in variable_chunks:
        variables = variable_chunk.items()
        for variable in variables:
            fields_dict = variable[1]
            for field in fields_dict:
                if field not in target_fields:
                    target_fields.append(field)

    columns = ['id'] + target_fields

    corpus_df = pd.DataFrame(columns=columns)

    variable_chunks = list(corpus_dict.keys())

    relevant_variable_chunks = [value for value in variable_chunks if value in set(research_data)]

    for variable_chunk in relevant_variable_chunks:
        variables = corpus_dict[variable_chunk].items()
        for variable in variables:
            this_row = []
            id = variable[0]
            this_row = this_row + [id]
            fields_dict = variable[1]
            for column in columns[1:]:
                if column in fields_dict.keys():
                    this_row = this_row + [fields_dict[column]]
                else:
                    this_row = this_row + [" "]
            target_df = pd.DataFrame([this_row], columns=columns)
            corpus_df = pd.concat([corpus_df, target_df])

    corpus_df.to_csv(targets_path_en_df, sep='\t', header=True, index=False)


if __name__ == "__main__":
    run()