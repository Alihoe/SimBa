from scipy.stats import spearmanr


def analyse_correlation(embeddings, correlation_method):
        a = embeddings.shape[0]
        b = embeddings.shape[1]
        c = embeddings.shape[2]
        embeddings = embeddings.reshape(a, b*c)
        if correlation_method == "spearmanr":
            correlation, p_value = spearmanr(embeddings, axis=1, nan_policy='propagate')
            print(correlation)