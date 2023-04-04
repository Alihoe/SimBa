import os

from scipy.stats import spearmanr

base_path = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(base_path, "../../data/")


def analyse_feature_correlation(all_features, feature_scores, data_name):
    correlation, p_value = spearmanr(feature_scores, nan_policy='propagate')

    with open(DATA_PATH + data_name + "/" + data_name + "_correlation.txt", 'w') as f:
        print(all_features, file=f)
        print(correlation, file=f)
        for idx, feature in enumerate(all_features):
            correlations = correlation[idx]
            feature_correlation = dict(zip(all_features, correlations))
            feature_correlation_sorted = dict(sorted(feature_correlation.items(), key=operator.itemgetter(1), reverse=True))
            feature_correlation = dict(list(feature_correlation_sorted.items())[1:])
            print('-------------------------------------------', file=f)
            print('The correlation for feature ' + str(feature), file=f)
            for key, value in feature_correlation.items():
                print('with feature ' + str(key) + ' is ' + str(round(value, 3)), file=f)