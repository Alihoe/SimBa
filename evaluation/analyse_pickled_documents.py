import pickle

from evaluation import DATA_PATH
from src.utils import decompress_file, load_pickled_object

data_name = "75302_labels"

zipped_pkl_file = DATA_PATH + "cache/" + data_name + "/queries_spacy_ne_similarity.pickle.zip"
pkl_file = decompress_file(zipped_pkl_file)
queries_dict = load_pickled_object(pkl_file)

print(queries_dict["2"])
print(queries_dict["3"])

zipped_pkl_file = DATA_PATH + "cache/gesis_unsup_labels/targets_spacy_ne_similarity.pickle.zip"
pkl_file = decompress_file(zipped_pkl_file)

targets_dict = load_pickled_object(pkl_file)

print(targets_dict["exploredata-ZA5564_VarV29"])
print(targets_dict["exploredata-ZA2391_Varv42"])
print(targets_dict["exploredata-ZA2563_Varv130"])