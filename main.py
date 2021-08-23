from tomek import removeTomekLinks, detectTomekLinks
from utils import make_cluster_data

import matplotlib.pyplot as plt


def run():
    X, y = make_cluster_data(centerbox=(-6.0, 6.0))
    plt.scatter(X[:,0],X[:,1], c = y)
    plt.title("Clusters")

    plt.show()

    nonlinks, tomeklinks = detectTomekLinks(X, y)
    X_clean, y_clean = removeTomekLinks(X, y, tomeklinks)
    plt.scatter(X_clean[:,0],X_clean[:,1], c = y_clean)
    plt.title("Cleaned")

    plt.show()



if __name__ == '__main__':
    run()
