import requests
from zipfile import ZipFile
from pathlib import Path
from os.path import join
from os import listdir, rmdir
from shutil import move

from src.get_data import DATA_PATH


def run():
    directory = DATA_PATH + "clef_2022_checkthat_2b_english/training_data"
    Path(directory).mkdir(parents=True, exist_ok=True)
    queries = requests.get('https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task2/data/subtask-2b--english/CT2022-Task2B-EN-Train-Dev_Queries.tsv')
    with open(directory+"/queries.tsv", 'wb') as f:
        f.write(queries.content)
    dev_qrels = requests.get('https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task2/data/subtask-2b--english/CT2022-Task2B-EN-Dev_QRELs.tsv')
    with open(directory+"/dev_qrels.tsv", 'wb') as f:
        f.write(dev_qrels.content)
    train_qrels = requests.get('https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task2/data/subtask-2b--english/CT2022-Task2B-EN-Train_QRELs.tsv')
    with open(directory+"/train_qrels.tsv", 'wb') as f:
        f.write(train_qrels.content)


if __name__ == "__main__":
    run()