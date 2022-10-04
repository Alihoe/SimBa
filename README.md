# SimBa

**++++++++++++Still under construction++++++++++++**

SimBa is an unsupervised IR-pipeline designed for STS tasks. 
For Candidate Retrieval it makes use of sentence embedding models,
for Re-Ranking additionally of simple lexical overlap between query and target.

There are separate scripts available for getting CLEF CheckThat! claim matching datasets,
for candidate retrieval, for re-ranking and for evaluation.

SimBa is an unsupervised IR-pipeline designed for Semantic Textual Similarity(STS) tasks.
Its original version was presented in REF and slightly edited afterwards, concentrating
on the best working features for the unsupervised approach.
For Candidate Retrieval it encodes queries and targets using sentence embedding models
and retrieves the closest target for every query based on their embeddings' spatial distances.
The candidates are selected based on the union of the k closest targets for every model per query.
Re-Ranking is also based on sentence embedding distances, as well as
simple lexical overlap between query and target. 
Therefore the spatial distance scores and the ratio of lexical overlap are simply averaged. 
Any number of sentence embeddings models, as well as any spatial distance measure can be used
for both retrieval and re-ranking.
The results presented in this paper were created using the sentence encoders "all-mpnet-base-v2"
to retrieve the k=50 closest candidate-targets for every input query according to braycurtis distance.
For re-ranking we used "all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large",
"sentence-transformers/sentence-t5-base",
"https://tfhub.dev/google/universal-sentence-encoder/4"] and the ratio of similar words as features.


## Performance on CLEF CheckThat! claim matching datasets

CLEF CheckThat! claim matching datasets 

| Datast  | Map@5 |  
|---|---|
| 2020 2a English  | 0.9525 |    
| 2021 2a English  | 0.9035 |      
| 2021 2b English  | 0.4662 |  
| 2022 2a English  | 0.9337 |    
| 2022 2b English  | 0.5282 | 

All MAP scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. [0.3076923076923077, 0.35256410256410253, 0.36102564102564105, 0.3694200244200244, 0.37178688832534984, 0.3731780700878844, 0.374520842851623]
INFO : MRR score 0.3833437594326711
INFO : All P scores on threshold from [1, 3, 5, 10, 20, 50, 1000]. [0.3230769230769231, 0.15384615384615383, 0.09846153846153845, 0.05846153846153847, 0.03076923076923077, 0.013230769230769232, 0.0009692307692307692]



For these results I used the sentence encoders "all-mpnet-base-v2" to retrieve the k=50 closest(braycurtis distance)candidate-targets for every input query. For re-ranking I used ["all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "https://tfhub.dev/google/universal-sentence-encoder/4"] and ["similar_words_ratio"] with k=5.

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

## Experimenting with retrieval parameters

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

k=50 --> Seems to be the optimal value

for dataset clef_2022_checkthat_2a_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|**0.9809**|0.9310|0.9761|
|all but Infersent|**0.9809**|0.9310| 0.9761
|all-mpnet-base-v2|0.9761|**0.9337**|0.9761|

for dataset clef_2022_checkthat_2b_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|**0.8072**|0.5133|0.7229|
|all but Infersent|0.7952|0.5133|0.7108 
|all-mpnet-base-v2|0.6386|**0.5282**|0.6386

k=20 

for dataset clef_2022_checkthat_2a_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|0.9761|0.9310|0.9761|
|all but Infersent|0.9761|0.9310|0.9761|
|all-mpnet-base-v2|0.9617|0.9268|0.9617|

for dataset clef_2022_checkthat_2b_english

|Used Models|Recall after retrieval| Map@5 after re-ranking|Recall after re-ranking|
|-----------|----------------------|-----------------------|----------------------|
|all models|0.7108|0.5133|0.7108|
|all but Infersent|0.7108|0.5133|0.7108|
|all-mpnet-base-v2|0.5663|0.4923|0.5663|

### Best parameters for retrieval

k = 50\
use only all-mpnet-base-v2

## Experimenting with different models for re-ranking 

for dataset clef_2022_checkthat_2a_english

|used models| MAP@5|
|-----------|------|
|all models|0.9130|
|all but Infersent|**0.9337**|
|all-mpnet-base-v2|0.9148|
|princeton-nlp/sup-simcse-roberta-large|0.8969|
|sentence-transformers/sentence-t5-base|0.8774|
|infersent|0.7066|
|https://tfhub.dev/google/universal-sentence-encoder/4|0.8537|
|all-mpnet-base-v2, ..sentence-t5-base, ..simcse..|*0.9312*|
|all-mpnet-base-v2, ..sentence-t5-base|0.9254|
|all-mpnet-base-v2, ..simcse..|0.9258|
|..sentence-t5-base, ..simcse..|0.9091|

for dataset clef_2022_checkthat_2b_english

|used models| MAP@5|
|-----------|------|
|all models|**0.5454**|
|all but Infersent|0.5282|
|all-mpnet-base-v2|0.5244|
|princeton-nlp/sup-simcse-roberta-large|0.5179|
|sentence-transformers/sentence-t5-base|0.5288|
|infersent|0.4936|
|https://tfhub.dev/google/universal-sentence-encoder/4|0.5295|
|all-mpnet-base-v2, ..sentence-t5-base, ..simcse..|0.5269|
|all-mpnet-base-v2, ..sentence-t5-base|*0.5423*|
|all-mpnet-base-v2, ..simcse..|0.5295|
|..sentence-t5-base, ..simcse..|0.5321|

for dataset clef_2022_checkthat_2a_english

without lexical information

|used models without lexical information| MAP@5|
|-----------|------|
|all models|0.8980|
|all but Infersent|0.9190|
|all-mpnet-base-v2|0.8861|
|all-mpnet-base-v2, ..sentence-t5-base, ..simcse..|0.9206|
|all-mpnet-base-v2, ..sentence-t5-base|0.9075|
|

for dataset clef_2022_checkthat_2b_english

|used models without lexical information| MAP@5|
|-----------|------|
|all models|0.5077|
|all but Infersent|0.4885|
|all-mpnet-base-v2|0.3803|
|all-mpnet-base-v2, ..sentence-t5-base, ..simcse..|0.4477|
|all-mpnet-base-v2, ..sentence-t5-base|0.4336|

### Best Parameters for Re-Ranking

use all models but Infersent
use lexical information

## References
<a id="1">[1]</a> 
Conneau et al.: 
Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
https://arxiv.org/abs/1705.02364

<a id="2">[2]</a> 
https://github.com/facebookresearch/InferSent


