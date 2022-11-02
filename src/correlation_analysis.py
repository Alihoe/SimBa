from scipy.stats import spearmanr


def analyse_correlation(all_features, embeddings, correlation_method, data_name):
        all_feature_names = '_'.join(all_features)
        a = embeddings.shape[0]
        b = embeddings.shape[1]
        c = embeddings.shape[2]
        embeddings = embeddings.reshape(a, b*c)
        if correlation_method == "spearmanr":
            correlation, p_value = spearmanr(embeddings, axis=1, nan_policy='propagate')
            print(correlation)
        with open('../data/evaluation/' + all_feature_names + '_' + data_name + '.txt', 'w') as f:
            print(all_feature_names)
            print(correlation, file=f)