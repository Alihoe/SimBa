import numpy as np


def predict_n_closest_cluster(data, classifier, n=2):
    dist_centers = classifier.transform(data)
    sorted_dist_centers = np.asmatrix(np.argsort(dist_centers, axis=1)).transpose()
    return np.ravel(sorted_dist_centers[n-1])

