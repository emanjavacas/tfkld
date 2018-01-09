
import numpy as np
from scipy import sparse


def _get_counts(p1, p2, labels):
    nrows, ndim = p1.shape
    counts = np.ones((4, ndim))

    for row in range(nrows):
        p1row, p2row = sparse.find(p1[row])[1], sparse.find(p2[row])[1]
        non_shared_idx = np.union1d(
            np.setdiff1d(p1row, p2row), np.setdiff1d(p2row, p1row))
        shared_idx = np.intersect1d(p1row, p2row)

        if labels[row] == 0:
            # 0: non-shared negative
            counts[0, non_shared_idx] += 1.0
            # 1: shared negative
            counts[1, shared_idx] += 1.0
        elif labels[row] == 1:
            # 2: non-shared positive
            counts[2, non_shared_idx] += 1.0
            # 3: shared positive
            counts[3, shared_idx] += 1.0

    return counts


def _kld(counts, smoothing):
    # normalize counts to probabilities by label
    counts += smoothing
    probs = counts / np.vstack(map(sum, np.split(counts, 2))).repeat(2, 0)
    # compute KL-d: sum(P(i) * log[P(i) / Q(i)])
    P, Q = np.split(probs, 2)
    return (P * np.log((P / Q) + 1e-7)).sum(axis=0)


class TFKLD(object):
    """
    sklearn: TFKLD
    """
    def __init__(self, smoothing=0.05):
        self.weight = None
        self.smoothing = smoothing

    def fit(self, p1, p2, labels):
        if isinstance(labels, list) or isinstance(labels, tuple):
            labels = np.array(labels)
        if p2.shape != p1.shape or p1.shape[0] != labels.shape[0]:
            raise ValueError("Input matrices must be equal shape")

        counts = _get_counts(p1, p2, labels)
        self.weight = _kld(counts, self.smoothing)

        return self

    def transform(self, m):
        if len(m.shape) != 2 or m.shape[1] != self.weight.shape[0]:
            raise ValueError("Expected 2D matrix: N x {}".format.self.weight.shape[0])

        return m.multiply(self.weight[None, :]).tocsr()

    def fit_transform(self, p1, p2, labels):
        self.fit(p1, p2, labels)
        return self.transform(p1), self.transform(p2)
