
# SV IDENT Variable Disambiguation

## Trial Data

### English only

Test data: 23 queries, 48 qrels\
Training data: 101 queries, 143 qrels\
all data: 124 queries, 191 qrels

#### All variable fields

|Set Up Retrieval|Set Up Re-Ranking|MAP@10 Test Data|MAP@10 Train Data|
|----------------|-----------------|----------------|-----------------|
|"all-mpnet-base-v2", braycurtis distance|"all-mpnet-base-v2", "princeton-nlp/sup-simcse-roberta-large", "sentence-transformers/sentence-t5-base", "https://tfhub.dev/google/universal-sentence-encoder/4"], ratio of similar words|0.7582|0.6389