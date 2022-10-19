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
    data_name = "sv_ident_trial_de_test_and_train"

    Path("../../data/"+data_name).mkdir(parents=True, exist_ok=True)
    general_path = "../../data/"+data_name
    queries_path_de_train = general_path + "/queries_de_train.tsv"
    queries_path_de_test = general_path + "/queries_de_test.tsv"

    queries_path_de = general_path + "/queries.tsv"
    qrels_path_de = general_path + "/gold.tsv"

    queries_qrels_de_test = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/test/de.tsv")
    queries_qrels_de_train = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/train/de.tsv")

    with open(queries_path_de_test, 'wb') as f:
        f.write(queries_qrels_de_test.content)

    with open(queries_path_de_train, 'wb') as f:
        f.write(queries_qrels_de_train.content)

    df_de_test = pd.read_csv(queries_path_de_test, sep='\t')
    df_de_train = pd.read_csv(queries_path_de_train, sep='\t')
    df_de = pd.concat([df_de_test, df_de_train])
    df_de = df_de.loc[df_de['is_variable'] != 0]

    queries_df_de =  pd.DataFrame(columns=['uuid', 'text'])
    qrels_df_de = pd.DataFrame(columns=['uuid', '0', 'variable', '1'])

    for index, row in df_de.iterrows():
        uuid = row['uuid']
        text = row['text']
        variables = row['variable'].split(",")
        for variable in variables:
            has_variable, id = get_variable_names(variable)
            if has_variable:
                qrels_df = pd.DataFrame([[uuid, 0, id, 1]], columns=['uuid', '0', 'variable', '1'])
                qrels_df_de = pd.concat([qrels_df_de, qrels_df])
                queries_df = pd.DataFrame([[uuid, text]], columns=['uuid', 'text'])
                queries_df_de = pd.concat([queries_df_de, queries_df])

    queries_df_de = queries_df_de.drop_duplicates()

    qrels_df_de.to_csv(qrels_path_de, sep= '\t', header = False, index=False)
    queries_df_de.to_csv(queries_path_de, sep= '\t', header = False, index=False)

    targets_path_de = general_path + "/corpus"
    targets_de = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/vocabulary/de.tsv")
    with open(targets_path_de, 'wb') as f:
        f.write(targets_de.content)

    os.remove(general_path + "/queries_de_train.tsv")
    os.remove(general_path + "/queries_de_test.tsv")


if __name__ == "__main__":
    run()