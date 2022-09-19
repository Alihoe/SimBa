import os
import shutil

import requests
from zipfile import ZipFile
from pathlib import Path
from os.path import join
from os import listdir, rmdir
from shutil import move

from src.utils import delete_first_line_of_tsv


def run():

    Path("../../data/clef_2021_checkthat_2a_arabic").mkdir(parents=True, exist_ok=True)
    general_path = "../../data/clef_2021_checkthat_2a_arabic"
    queries_path = general_path + "/queries.tsv"
    queries = requests.get("https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/raw/master/task2/test-input/subtask-2a--arabic.zip")
    with open(general_path+"/data.zip", 'wb') as f:
        f.write(queries.content)
    with ZipFile(general_path+"/data.zip", 'r') as zipObj:
        for file in zipObj.namelist():
            if file.startswith('subtask'):
                zipObj.extract(file, general_path)
    os.rename(general_path + "/subtask-2a--arabic/CT2021-Task2A-AR-Test_Queries.tsv", queries_path)
    delete_first_line_of_tsv(queries_path)
    shutil.rmtree(general_path + "/subtask-2a--arabic", ignore_errors=False, onerror=None)
    os.remove(general_path+"/data.zip")

    corpus = requests.get("https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/raw/master/task2/data/subtask-2a--arabic/v1.5/vclaims.zip")
    corpus_filepath = "../../data/clef_2021_checkthat_2a_arabic/corpus"
    with open(corpus_filepath+".zip", 'wb') as f:
        f.write(corpus.content)
    with ZipFile(corpus_filepath+".zip", 'r') as zipObj:
        for file in zipObj.namelist():
            zipObj.extract(file, corpus_filepath)
    for filename in listdir(corpus_filepath+'/vclaims'):
        move(join(corpus_filepath+'/vclaims/', filename), join(corpus_filepath, filename))
    shutil.rmtree(corpus_filepath+'/vclaims', ignore_errors=False, onerror=None)
    os.remove(general_path + "/corpus.zip")

    gold_file_path = "../../data/clef_2021_checkthat_2a_arabic/gold.tsv"
    gold_file = requests.get("https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/raw/master/task2/test-gold/subtask-2a--arabic.zip")
    with open(general_path+"/data.zip", 'wb') as f:
        f.write(gold_file.content)
    with ZipFile(general_path+"/data.zip", 'r') as zipObj:
        for file in zipObj.namelist():
            if file.startswith('subtask'):
                zipObj.extract(file, general_path)
    os.rename(general_path + "/subtask-2a--arabic/CT2021-Task2A-AR-Test_QRELs.tsv", gold_file_path)
    shutil.rmtree(general_path + "/subtask-2a--arabic", ignore_errors=False, onerror=None)
    os.remove(general_path+"/data.zip")


if __name__ == "__main__":
    run()