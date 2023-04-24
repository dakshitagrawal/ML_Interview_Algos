from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans


def get_labels(centers, points):

    """
    Get distances between each point and each cluster center
    :param centers: Cluster centers of shape (k, d)
    :param points: Data points of shape (n, d)
    """
    dists = ((points[:, None] - centers[None]) ** 2).sum(-1)
    return dists.argmin(-1)


def kmeans(X, k):
    """
    K-Means

    :param X: An (n, d) numpy array. Each row is a sample with d dimensions
    :param k: Number of clusters
    :return centers: A (k, d) numpy array. Each row is a center with d dimensions.
    :return labels: A (n,) numpy array indicating which cluster each sample belongs to.

    IMPORTANT note:
        Use the first k rows of X as the initial centers.
        Since the result depends on the initial centers, you will get
        wrong answers if you fail to do this.
    """

    centers = X[:k]
    prev_labels = None
    labels = get_labels(centers, X)
    iterations = 0

    # loop till labels don't change
    while prev_labels is None or (prev_labels != labels).any():
        # update cluster centers
        centers = np.array([X[np.nonzero(labels == i)].mean(0) for i in range(k)])

        # get labels
        prev_labels = labels.copy()
        labels = get_labels(centers, X)
        iterations += 1

    print(iterations)
    return centers, labels


def sklearn_kmeans(X, k):
    """
    K-Means implementation should return clusters close to this function.
    """
    km = KMeans(
        n_clusters=k, init=lambda x, n, random_state: x[:n, :], n_init=1, tol=1e-6
    )
    km.fit(X)
    return km.cluster_centers_, km.labels_


def main(args):
    n = args.n
    k = args.k

    # Load data
    data = pd.read_csv("adult.csv", header=None)
    data = data[[0, 2, 4]]
    X = data.to_numpy()
    X = X[:n].astype(int)

    X_mean = X.mean(0, keepdims=True)
    X_std = X.std(0, keepdims=True)
    X_norm = (X - X_mean) / X_std

    # Run k-means
    X0 = X_norm.copy()
    a, b = kmeans(X_norm, k)
    a0, b0 = sklearn_kmeans(X0, k)

    print("Your clusters:", b)
    print("sklearn clusters:", b0)
    print(a)

    print("ARI = ", adjusted_rand_score(b, b0))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, help="number of data points", default=30000)
    parser.add_argument("--k", type=int, help="number of clusters", default=10)
    args = parser.parse_args()

    main(args)
