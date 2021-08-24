import numpy as np
from sklearn.neighbors import NearestNeighbors


def detect(X, y, k=2):
    if k < 2:
        raise ValueError("k must be >=2")

    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nn.fit(X)

    # k_neighbours stores the k nearest neighbour
    distances, indices = nn.kneighbors(X)

    # k_tomek stores the ones where the labels conflict.
    # the first index corresponds to the point itself,
    # Because the query set matches the training set,
    # the nearest neighbor of each point is the point itself, at a distance of zero.
    k_tomek = indices[y != y[indices[:, k - 1]]]

    # return indices
    tomek_links = np.unique(k_tomek[:, k - 1])

    return tomek_links


def get_non_links(X, tomek_links):
    index = np.arange(0, len(X))
    non_links = set(index) - set(tomek_links)
    #return indices
    return np.asarray(list(non_links))


def remove(X, y, tomek_links):
    return np.delete(X, tomek_links,axis=0), np.delete(np.array(y), tomek_links, axis=0)

