import requests
from zipfile import ZipFile
from pathlib import Path
from os.path import join
from os import listdir, rmdir
from shutil import move


def run():
    dir = "../../data/clef_2022_checkthat_arabic"
    Path(dir).mkdir(parents=True, exist_ok=True)
    queries = requests.get('https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task2/data/subtask-2a--arabic/test/CT2022-Task2A-AR-Test_Queries.tsv')
    with open(dir+"/queries.tsv", 'wb') as f:
        f.write(queries.content)
    corpus = requests.get('https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task2/data/subtask-2a--arabic/vclaims_json.zip')
    corpus_filepath = dir+"/corpus"
    with open(corpus_filepath+".zip", 'wb') as f:
        f.write(corpus.content)
    with ZipFile(corpus_filepath+".zip", 'r') as zipObj:
        for file in zipObj.namelist():
            if file.startswith('vclaims'):
                zipObj.extract(file, corpus_filepath)
    for filename in listdir(join(corpus_filepath, 'vclaims')):
        move(join(corpus_filepath, 'vclaims', filename), join(corpus_filepath, filename))
    rmdir(join(corpus_filepath, 'vclaims'))
    gold_file = requests.get("https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task2/data/subtask-2a--arabic/test/CT2022-Task2A-AR-Test_Qrels_gold.tsv")
    with open(dir+"/gold.tsv", 'wb') as f:
        f.write(gold_file.content)


if __name__ == "__main__":
    run()