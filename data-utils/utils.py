from sklearn.datasets import make_blobs


def make_cluster_data(n_samples=1000, centers=2, n_features=2, centerbox=(-10.0, 10.0)):
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, center_box=centerbox)
    return X, y
