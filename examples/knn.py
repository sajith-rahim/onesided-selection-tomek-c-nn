from utils import make_cluster_data
from kNN.KNN import KNN


def run():
    X, y = make_cluster_data(centerbox=(-4.0, 4.0))
    knn = KNN(X, y, k=3)
    res = knn.classify(X[99])
    print(res)


if __name__ == '__main__':
    run()
