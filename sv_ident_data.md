
# SV IDENT Variable Disambiguation

**1. Reproduce paper results** (https://aclanthology.org/2022.sdp-1.30/)

**2. Incorporate multilingual sentence embedding models**

**3. Analyse Data in Detail**

**4. Data Augmentation**

## 

## Trial Data

### English only

Test data: 23 queries, 48 qrels\
Training data: 101 queries, 143 qrels\
all data: 124 queries, 191 qrels

#### All variable fields

|Set Up Retrieval|Set Up Re-Ranking|MAP@10 Test Data|MAP@10 Train Data|MAP@10|
|----------------|-----------------|----------------|-----------------|------|
|"all-mpnet-base-v2", braycurtis distance|"all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "https://tfhub.dev/google/universal-sentence-encoder/4"], ratio of similar words|0.7582|0.6389|0.6610|

## Train and Val Data

### English only

#### All variable fields

|Set Up Retrieval|Set Up Re-Ranking|MAP@10|
|----------------|-----------------|------|
|"all-mpnet-base-v2", braycurtis distance|"all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "https://tfhub.dev/google/universal-sentence-encoder/4"], ratio of similar words|0.1166|

#### Certain Variable Fields

|Variable Fields|MAP@10|
|---------------|------|
|question text  |0.0770|
|variable_label_topic_en_question_text_question_text_en|0.1279|

## English and German


Set Up Retrieval and Re-Ranking|Variable Fields|MAP@10|
|------------------------------|---------------|------|
|eng: Sentence T5              |eng: variable_label_topic_en_question_text_question_text_en|  |
|de: Sahajtomar/German-semantic|de: all but English                |0.0545|
|multilingual: distiluse-base-multilingual-cased-v1|all|0.0379|

