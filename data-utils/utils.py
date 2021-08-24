from sklearn.datasets import make_blobs
import numpy as np


def make_cluster_data(n_samples=1000, centers=2, n_features=2, centerbox=(-7.0, 7.0)):
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, center_box=centerbox)
    return X, y


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))

    similarity = np.dot(a, b.T) / (a_norm * b_norm)
    dist = 1.0 - similarity
    return dist
