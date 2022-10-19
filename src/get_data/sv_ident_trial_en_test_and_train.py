import os
from os.path import join
from os import listdir, rmdir
from shutil import move

import shutil

from zipfile import ZipFile
from pathlib import Path

import pandas as pd
import requests

from src.utils import delete_first_line_of_tsv


def get_variable_names(sequence):
    splitted = sequence.split("-")
    variable = splitted[0]
    relevance = splitted[1]
    if relevance == "Yes":
        return True, "v"+variable
    else:
        return False, variable



def run():

    # queries
    # qrels/gold
    # targets
    data_name = "sv_ident_trial_en_test_and_train"

    Path("../../data/"+data_name).mkdir(parents=True, exist_ok=True)
    general_path = "../../data/"+data_name
    queries_path_en_train = general_path + "/queries_en_train.tsv"
    queries_path_en_test = general_path + "/queries_en_test.tsv"

    queries_path_en = general_path + "/queries.tsv"
    qrels_path_en = general_path + "/gold.tsv"

    queries_qrels_en_test = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/test/en.tsv")
    queries_qrels_en_train = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/train/en.tsv")

    with open(queries_path_en_test, 'wb') as f:
        f.write(queries_qrels_en_test.content)

    with open(queries_path_en_train, 'wb') as f:
        f.write(queries_qrels_en_train.content)

    df_en_test = pd.read_csv(queries_path_en_test, sep='\t')
    df_en_train = pd.read_csv(queries_path_en_train, sep='\t')
    df_en = pd.concat([df_en_test, df_en_train])
    df_en = df_en.loc[df_en['is_variable'] != 0]

    queries_df_en =  pd.DataFrame(columns=['uuid', 'text'])
    qrels_df_en = pd.DataFrame(columns=['uuid', '0', 'variable', '1'])

    for index, row in df_en.iterrows():
        uuid = row['uuid']
        text = row['text']
        variables = row['variable'].split(",")
        for variable in variables:
            has_variable, id = get_variable_names(variable)
            if has_variable:
                qrels_df = pd.DataFrame([[uuid, 0, id, 1]], columns=['uuid', '0', 'variable', '1'])
                qrels_df_en = pd.concat([qrels_df_en, qrels_df])
                queries_df = pd.DataFrame([[uuid, text]], columns=['uuid', 'text'])
                queries_df_en = pd.concat([queries_df_en, queries_df])

    queries_df_en = queries_df_en.drop_duplicates()

    qrels_df_en.to_csv(qrels_path_en, sep= '\t', header = False, index=False)
    queries_df_en.to_csv(queries_path_en, sep= '\t', header = False, index=False)

    targets_path_en = general_path + "/corpus"
    targets_en = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/vocabulary/en.tsv")
    with open(targets_path_en, 'wb') as f:
        f.write(targets_en.content)

    os.remove(general_path + "/queries_en_train.tsv")
    os.remove(general_path + "/queries_en_test.tsv")


if __name__ == "__main__":
    run()