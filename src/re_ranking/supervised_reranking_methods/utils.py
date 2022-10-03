import numpy as np
import pandas as pd


def prepare_binary_dataset(queries_and_features_dictionary, target_ids, gold_qrels_path):

    df = pd.DataFrame(columns=['query_id', 'target_id', 'features', 'label'])
    queries = []
    targets = []
    all_features = []

    for query, features in queries_and_features_dictionary.items():
        number_targets = len(target_ids[query])
        features_len = len(features)
        for i in range(number_targets):
            queries.append(query)
            targets.append(target_ids[query][i])
            curr_features = []
            for ix in range(features_len):
                if type(features[ix]) == np.ndarray:
                    curr_features.append(features[ix][0][i])
                else:
                    curr_features.append(features[ix][i])
            all_features.append(curr_features)

    df['query_id'] = queries
    df['target_id'] = targets
    df['features'] = all_features
    df['label'] = 0

    gold_df = pd.read_csv(gold_qrels_path, sep='\t', names=['query_id', '0', 'target_id', '1'], dtype=str)
    gold_query_ids = gold_df['query_id'].tolist()
    gold_target_ids = gold_df['target_id'].tolist()

    for i in range(len(gold_query_ids)):
        df.loc[(df['query_id']==gold_query_ids[i]) & (df['target_id']==gold_target_ids[i]),'label']= 1

    return df





