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
    data_name = "sv_ident_trial"

    Path("../../data/"+data_name).mkdir(parents=True, exist_ok=True)
    general_path = "../../data/"+data_name
    queries_path = general_path + "/queries.tsv"
    qrels_path = general_path + "/gold.tsv"
    queries_path_en = general_path + "/en_queries.tsv"
    queries_path_de = general_path + "/de_queries.tsv"

    queries_qrels_en = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/test/en.tsv")
    queries_qrels_de = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/test/de.tsv")

    with open(queries_path_en, 'wb') as f:
        f.write(queries_qrels_en.content)

    with open(queries_path_de, 'wb') as f:
        f.write(queries_qrels_de.content)

    df_en = pd.read_csv(queries_path_en, sep='\t')
    df_de = pd.read_csv(queries_path_de, sep='\t')

    df = pd.concat([df_en, df_de])

    df = df.loc[df['is_variable'] != 0]

    queries_df =  pd.DataFrame(columns=['uuid', 'text'])
    qrels_df = pd.DataFrame(columns=['uuid', '0', 'variable', '1'])

    for index, row in df.iterrows():
        uuid = row['uuid']
        text = row['text']
        variables = row['variable'].split(",")
        for variable in variables:
            has_variable, id = get_variable_names(variable)
            if has_variable:
                qrels_row = pd.DataFrame([[uuid, 0, id, 1]], columns=['uuid', '0', 'variable', '1'])
                qrels_df = pd.concat([qrels_df, qrels_row])
                queries_row = pd.DataFrame([[uuid, text]], columns=['uuid', 'text'])
                queries_df = pd.concat([queries_df, queries_row])

    queries_df = queries_df.drop_duplicates()

    qrels_df.to_csv(qrels_path, sep= '\t', header = False, index=False)
    queries_df.to_csv(queries_path, sep= '\t', header = False, index=False)

    targets_path_en = general_path + "/corpus_en"
    targets_en = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/vocabulary/en.tsv")
    with open(targets_path_en, 'wb') as f:
        f.write(targets_en.content)

    targets_path_de = general_path + "/corpus_de"
    targets_de = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/vocabulary/de.tsv")
    with open(targets_path_de, 'wb') as f:
        f.write(targets_de.content)

    df_en = pd.read_csv(targets_path_en, sep='\t')
    df_de = pd.read_csv(targets_path_de, sep='\t')

    df = pd.concat([df_en, df_de])

    targets_path = general_path + "/corpus"
    df.to_csv(targets_path, sep='\t', header=True, index=False)

    os.remove(queries_path_en)
    os.remove(queries_path_de)
    os.remove(targets_path_en)
    os.remove(targets_path_de)


if __name__ == "__main__":
    run()