
import warnings
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


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
            counts[1, shared_idx ]+= 1.0
        elif labels[row] == 1:
            # 2: non-shared positive
            counts[2, non_shared_idx] += 1.0
            # 3: shared positive
            counts[3, shared_idx] += 1.0

    return counts


def _kld(counts, smoothing):
    # add smoothing to the counts (avoid zero entries)
    counts += smoothing
    # normalize counts to probabilities by label
    pos, neg = np.split(counts, 2)
    probs = counts / np.vstack(map(sum, [pos, neg])).repeat(2, 0)
    # compute KL-d: sum(P(i) * log[P(i) / Q(i)])
    P, Q = np.split(probs, 2)

    return (P * np.log((P / Q) + 1e-7)).sum(axis=0)


def _apply_weight(weight, m):
    return m.multiply(weight[None, :]).tocsr()


class TFKLD(CountVectorizer):
    """
    sklearn: TFKLD subclass of CountVectorizer
    """
    def __init__(self, dtype=np.float, **kwargs):
        if 'dtype' in kwargs:
            warnings.warn("Ignoring `dtype` argument. Default to float.")
            del kwargs['dtype']
        super(TFKLD, self).__init__(**kwargs)

        self.weight = None

    def fit(self, p1, p2, labels, smoothing=0.05):
        # strings to counts
        super().fit_transform(p1 + p2)
        p1 = super().transform(p1)
        p2 = super().transform(p2)

        if isinstance(labels, list) or isinstance(labels, tuple):
            labels = np.array(labels)
        if p2.shape != p1.shape or p1.shape[0] != labels.shape[0]:
            raise ValueError("Input matrices must be equal shape")
        # get weight
        self.weight = _kld(_get_counts(p1, p2, labels), smoothing)  

        return self

    def transform(self, m):
        m = super().transform(m)

        if len(m.shape) != 2 or m.shape[1] != self.weight.shape[0]:
            raise ValueError("Expected 2D matrix: N x {}".format(self.weight.shape[0]))

        return _apply_weight(self.weight, m)

    def fit_transform(self, p1, p2, labels):
        # 
        self.fit(p1, p2, labels)

        return self.transform(p1), self.transform(p2)


class CosineClassifier(object):
    """
    sklearn: simple cosine similarity classifier
    """
    def __init__(self, step=1e-3):
        self.step = step
        self.threshold = None

    def _norm(self, m):
        return np.sqrt((m * m).sum(1))[:, None]

    def _sims(self, p1, p2):
        return ((p1 / self._norm(p1)) * (p2 / self._norm(p2))).sum(1)

    def _predict(self, sims, threshold):
        index = sims >= threshold
        preds = np.zeros_like(sims)
        preds[sims >= threshold] = 1.
        return preds

    def predict(self, p1, p2):
        if self.threshold is not None:
            return self._predict(self._sims(p1, p2), self.threshold)

        raise ValueError("Not fitted")

    def fit(self, p1, p2, labels):
        sims = self._sims()

        best_threshold, best_value = 0.0, 0
        for threshold in np.range(0, 1, self.step):
            value = (self._predict(sims, threshold) == labels).sum()
            if value > best_value:
                best_value = value
                best_threshold = threshold

        self.threshold = best_threshold


def dump_counts(counts):
    pass
