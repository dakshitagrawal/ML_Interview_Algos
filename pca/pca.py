from argparse import ArgumentParser

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA


def pca_fit(X_norm, k, p=None):
    """
    Apply PCA on normalized data X_norm and return PCA components that explains
    p% of total variance

    :param X: normalized data of type np.array (NxF)
    :param p: percentage of explained variance to capture
    :param k: number of principal components
    :return: return k PCA components that explain p% of total variance
    """

    U, D, Vt = np.linalg.svd(X_norm, full_matrices=False)

    explained_variance = D ** 2 / (len(X_norm) - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    if p is not None:
        cumulative_variance = np.cumsum(explained_variance_ratio)
        k = 0
        high = len(cumulative_variance) + 1

        while k < high:
            mid = (k + high) // 2

            if cumulative_variance[mid] < p:
                k = mid + 1
            else:
                high = mid

        k += 1

    pca_components = Vt[:k]
    explained_var = explained_variance_ratio[:k]
    return pca_components, explained_var


def pca_transform(X_norm, comp):
    """
    Transform data based on PCA components

    :param X: data of type np.array (N x F)
    :param comp: principal components (k x F)
    :return: transformed data
    """
    return X_norm @ comp.T


def pca_untransform(X_trans, comp):
    X_norm = X_trans @ comp
    return X_norm


def pca_sklearn(X, k):
    pca_sk = PCA(k, svd_solver="full")
    pca_sk.fit(X)
    return pca_sk


def main(args):
    X = datasets.load_iris().data

    X_mean = X.mean(0, keepdims=True)
    X_std = X.std(0, keepdims=True)
    X_norm = (X - X_mean) / X_std

    pca_components, pca_explained_var = pca_fit(X_norm, args.k, args.p)
    print(pca_components)
    print(pca_explained_var)

    pca_sk = pca_sklearn(X_norm, args.k)
    signs = np.where(pca_sk.components_[:, 0] == pca_components[:, 0], 1.0, -1.0)
    pca_components *= signs[:, None]
    assert (pca_sk.components_ - pca_components).sum() < 1e-6
    assert (pca_sk.explained_variance_ratio_ - pca_explained_var).sum() < 1e-6

    X_trans = pca_transform(X_norm, pca_components)
    X_trans_sk = pca_sk.fit_transform(X_norm)
    assert (X_trans - X_trans_sk).sum() < 1e-6

    assert (X_norm - pca_untransform(X_trans, pca_components)).sum() < 1e-6


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--k", type=int, help="number of principal components", required=True
    )
    parser.add_argument("--p", type=float)
    args = parser.parse_args()

    main(args)
