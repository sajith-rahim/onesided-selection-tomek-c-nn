from CondensedNN.IterativeCNN import IterativeCondensedNN
from tomek import remove, detect, get_non_links
from utils import make_cluster_data

import matplotlib.pyplot as plt

def run2():
    X, y = make_cluster_data(centerbox=(-4.0, 4.0))
    cnn = IterativeCondensedNN()
    cnn.fit(X, y)
def run():
    X, y = make_cluster_data(centerbox=(-4.0, 4.0))

    grid = plt.GridSpec(3, 4, wspace=0.4, hspace=0.3)
    plt.subplot(grid[0, 0:])
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Clusters")

    tomek_links = detect(X, y)
    non_links = get_non_links(X, y)
    plt.subplot(grid[1, 1:3])
    plt.scatter(X[tomek_links, 0], X[tomek_links, 1], c=y[tomek_links])
    plt.title("Tomek Links")

    X_clean, y_clean = remove(X, y, tomek_links)

    plt.subplot(grid[2, 0:])
    plt.scatter(X_clean[:, 0], X_clean[:, 1], c=y_clean)
    plt.title("Cleansed")

    plt.show()


if __name__ == '__main__':
    run2()
