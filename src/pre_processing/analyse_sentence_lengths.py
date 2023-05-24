import json
import os
import numpy as np


def analyse_sentence_length(sentence_extractions_path: str, percent: int = 95):
    all_jsons = os.listdir(sentence_extractions_path)
    sentence_lengths = []
    for c_json in all_jsons:
        with open(sentence_extractions_path + "/" + c_json, encoding='utf-8') as f:
            doc = json.load(f)
            doc_sentences = doc["sentences"]
            doc_sentence_lengths = [len(sentence) for sentence in doc_sentences]
            sentence_lengths.extend(doc_sentence_lengths)
    lengths = np.array(sentence_lengths)
    sentence_lengths.sort()
    percent_threshold = int(len(sentence_lengths) - (len(sentence_lengths)/100)*(100-percent))
    print("Approximately " + str(percent) + "% of the sentences are shorter than:")
    print(sentence_lengths[percent_threshold])
    print("The mean plus std of the sentence lengths is approximately:")
    print(int(np.mean(lengths) + np.std(lengths)))
    print("The mean plus two times the std of the sentence lengths is approximately:")
    print(int(np.mean(lengths) + 2*(np.std(lengths))))
    print("The mean plus half the std of the sentence lengths is approximately:")
    print(int(np.mean(lengths) + (np.std(lengths)/2)))


