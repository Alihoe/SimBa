# SimBa

SimBa is an unsupervised IR-pipeline designed for STS tasks. 
For Candidate Retrieval it makes use of sentence embedding models, for Re-Ranking additionally of simple lexical overlap between query and target.

There are separate scripts available for getting CLEF CheckThat! claim matching datasets,
for candidate retrieval, for re-ranking and for evaluation.

## Performance on CLEF CheckThat! claim matching datasets

CLEF CheckThat! claim matching datasets

| Datast  | Map@5 |  
|---|---|
| 2020 2a English  | 0.9567 |    
| 2021 2a English  | 0.9018 |      
| 2021 2b English  | 0.4635 |   
| 2022 2a English  | 0.9310 |     
| 2022 2b English  | 0.5133 |  

For these results I used the sentence encoders ["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"] and the union of the top k per feature as candidates with k=100. For re-ranking I used ["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "https://tfhub.dev/google/universal-sentence-encoder/4"] and ["similar_words_ratio"] with k=50.

## How to use

### Simple way: To use the pipeline for the already downloaded CLEF CheckThat! claim matching datasets with the same settings that produced the results shown in the table, simply use the "get_ranking_for_dataset.py" script and pass the dataset's name as an argument.

### Modify specific steps of the pipeline - Explanation of subscripts:

### 1. Getting CLEF CheckThat! claim matching datasets (https://sites.google.com/view/clef2022-checkthat)
Scripts can be found here: src/get_data

### 2. Candidate Retrieval
Script can be found here: src/candidate_retrieval/semantic_retrieval.py\

 Parameters:\
‘data’: name of data set to be used, different data sets should be stored in seperate folders in the “data” directory and consist of a queries-file, a gold-qrels-file  and a corpus (either a directory of jsons or a tsv-file)
→ Look at the example datasets for more information about format,
e.g. ‘clef_2022_checkthat_2a_english’

‘pre-processing’: optional, pre-processing scripts for the specific dataset can be stored in the “src/pre_processing” folder

‘sentence_embedding_models’: list of desired sentence embedding models to be considered, pass a list of sentence embedding models hosted by Huggingface or Tensorflow or simply pass "infersent" to use the infersent encoder (infersent model must be manually added into the infersent_encoder folder https://github.com/facebookresearch/InferSent ),
e.g. ["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"]

‘similarity_measure’: e.g. cosine or braycurtis (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

‘correlation_measure’: e.g. spearmanr (https://docs.scipy.org/doc/scipy/reference/stats.html), is not important yet

‘union_of_top_k_per_feature’: optional, if this is chosen candidate selection is based on the union of the top k matches per feature/model, otherwise the selection is based on the mean of the features

‘k’: top k matches to be considered for candidate selection

‘no_cache’: optional, if this is chosen sentence embedding will not be stored which can be impractical for experimenting with different parameters or using the same models for re-ranking (it takes a long time to encode all queries and especially all targets)

### 3. Re-Ranking
Script can be found here: src/re_ranking/multi_feature_re_ranking.py\

Parameters:\
‘data’: name of data set to be used, different data sets should be stored in seperate folders in the “data” directory and consist of a queries-file, a gold-qrels-file  and a corpus (either a directory of jsons or a tsv-file)
→ Look at the example datasets for more information about format,
e.g. ‘clef_2022_checkthat_2a_english’

‘sentence_embedding_models’: list of desired sentence embedding models to be considered, pass a list of sentence embedding models hosted by Huggingface or Tensorflow or simply pass "infersent" to use the infersent encoder (infersent model must be manually added into the infersent_encoder folder https://github.com/facebookresearch/InferSent ),
e.g. ["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"]

‘similarity_measure’: e.g. cosine or braycurtis (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

‘correlation_measure’: e.g. spearmanr (https://docs.scipy.org/doc/scipy/reference/stats.html)

‘lexical_similarity_measures’: right now only one is available ["similar_words_ratio"]

‘k’: top k matches to be considered for final ranking

‘no_cache’: optional, if this is chosen sentence embedding will not be stored which can be impractical for experimenting with different parameters (it takes a long time to encode all queries and especially all targets)

### 4. Evaluation
Script can be found here: evaluation/scorer/main.py\

Parameters:\
pred_qrels and gold_qrels
