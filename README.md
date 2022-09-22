# SimBa

**++++++++++++Still under construction++++++++++++**

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

## If you want to use Infersent:
Infersent[[1]](#1) was one of the first sentence encoder models and although outperformed on most baselines by more recent models, it can be of informational value on how sentence embedding models work since it relies on pre-trained word vectors.
To download the model and the corresponding word-vectors[[2]](#2) simply use the script "src/infersent_encoder/get_infersent_model_and_data.py".
The word-embeddings are very large though, and it takes a VERY long time to download them.

### Simple way:
To use the pipeline for the already downloaded CLEF CheckThat! claim matching datasets with the same settings that produced the results shown in the table, simply use the "get_ranking_for_dataset.py" script and pass the dataset's name as an argument.

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

## Reproducing results of SimBa for CheckThat! Lab 2022 as presented here: http://ceur-ws.org/Vol-3180/paper-40.pdf

2022 2b English supposed to be 0.4721

|Dataset        | Candidate Retrieval Parameters| Re-Ranking Parameters | Results (MAP@5) |
|---------------|-------------------------------|-----------------------|-----------------|
|2022 2b English|                               |                       | 0.4813          |
|SE-models      |["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"]| ["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"] | |
|distance       | cosine                        | cosine               |                 |
|k              | 50                            | 5                    |                 |
|---------------|-------------------------------|----------------------|-----------------|

2022 2a English supposed to be 0.9175, is 0.9230
2022 2b English supposed to be 0.4721, is 0.4813

## Experimenting with different sentence embedding models for retrieval

k = 100, union of top k features

for dataset clef_2022_checkthat_2a_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|**0.9952** |0.9310|0.9809|
|all but Infersent|**0.9952**|0.9310|0.9809|
|all-mpnet-base-v2|0.9904|**0.9328**|0.9809|
|princeton-nlp/sup-simcse-roberta-large|0.9617|0.9212|0.9617|
|sentence-transformers/sentence-t5-base|0.9809|0.9310|0.9761|
|infersent|0.4737|0.4737|0.4737|
|https://tfhub.dev/google/universal-sentence-encoder/4|0.9330|0.9031|0.9330|
--> Infersent not needed here
--> only using all-mpnet-base-v2 works best?

Correlation of sentence embedding models:

[[1.         0.52674273 0.56817182 0.2171586  0.46328655]
 [0.52674273 1.         0.56741003 0.27996202 0.42223214]
 [0.56817182 0.56741003 1.         0.3267913  0.4808523 ]
 [0.2171586  0.27996202 0.3267913  1.         0.403431  ]
 [0.46328655 0.42223214 0.4808523  0.403431   1.        ]]

for dataset clef_2022_checkthat_2b_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|**0.8554**|0.5317|0.7470|
|all but Infersent|**0.8554**|0.5133|0.7470|
|all-mpnet-base-v2|0.6988|**0.5448**|0.6988|
|princeton-nlp/sup-simcse-roberta-large|0.7952|0.5095|0.7711|
|sentence-transformers/sentence-t5-base|0.7108|0.5154|0.6867
|infersent|0.4337|0.4064|0.4337|
|https://tfhub.dev/google/universal-sentence-encoder/4|0.6265|0.5095|0.6145|

--> only using all-mpnet-base-v2 works best?

k=200 --> gets worse

for dataset clef_2022_checkthat_2a_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|**0.9952** |0.9310|0.9761|
|all but Infersent|**0.9952**|0.9310|0.9761|
|all-mpnet-base-v2|**0.9952**|**0.9328**|0.9856

for dataset clef_2022_checkthat_2b_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|**0.8675**|0.5133|0.7470|
|all but Infersent|**0.8675**|0.5133|0.7470
|all-mpnet-base-v2|0.7349|**0.5172**|0.7229

k=50 --> 

for dataset clef_2022_checkthat_2a_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|0.9809|0.9310|0.9761|
|all but Infersent|0.9809|0.9310| 0.9761
|all-mpnet-base-v2|0.9761|0.9337|0.9761|

for dataset clef_2022_checkthat_2b_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|0.8072|0.5133|0.7229|
|all but Infersent|0.7952|0.5133|0.7108 
|all-mpnet-base-v2|0.6386|0.5282|0.6386


## References
<a id="1">[1]</a> 
Conneau et al.: 
Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
https://arxiv.org/abs/1705.02364

<a id="2">[2]</a> 
https://github.com/facebookresearch/InferSent


