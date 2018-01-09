
import unittest
import random

import numpy as np
import lorem
from tqdm import tqdm

from tfkld import TFKLD, _get_counts, _kld, make_vectorizer


def _test_counts(p1, p2, labels):
    """
    Runs slow original python code
    """
    nrows, ncols = p1.shape
    count = np.ones((4, ncols))
    for row in tqdm(range(nrows)):
        label = labels[row]
        for d in range(ncols):
            if ((p1[row, d] > 0.) and (p2[row, d] == 0.)) \
               or ((p1[row, d] == 0.) and (p2[row, d] > 0.)):
                if label == 0:
                    count[0, d] += 1.0
                elif label == 1:
                    count[2, d] += 1.0
            elif (p1[row, d] > 0) and (p2[row, d] > 0):
                if label == 0:
                    count[1, d] += 1.0
                elif label == 1:
                    count[3, d] += 1.0
    return count


def _test_kld(counts, smoothing):
    # smoothing
    counts += smoothing
    # normalize
    pattern = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
    pattern = np.array(pattern)
    prob = counts / (pattern.dot(counts))
    # kl
    ratio = np.log((prob[0:2, :] / prob[2:4, :]) + 1e-7)
    weight = (ratio * prob[0:2, :]).sum(axis=0)
    return weight


class TestConditionals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p1, p2, labels = [], [], []
        for _ in range(100):
            p1.append(lorem.sentence())
            p2.append(lorem.sentence())
            labels.append(random.randint(0, 1))

        vec = make_vectorizer(p1 + p2)
        p1, p2 = vec.transform(p1), vec.transform(p2)
        labels = np.array(labels)
        cls.p1, cls.p2, cls.labels = p1, p2, labels

    def test_conditionals(self):
        p1, p2 = TestConditionals.p1, TestConditionals.p2
        labels = TestConditionals.labels

        counts1 = _get_counts(p1, p2, labels)
        counts2 = _test_counts(p1, p2, labels)
        self.assertTrue(np.all(counts1 == counts2))

        weight1 = _kld(counts1, 0.05)
        weight2 = _test_kld(counts2, 0.05)
        self.assertTrue(np.allclose(weight1, weight2))

        # transform using fast method
        tf = TFKLD()
        tf.weight = weight2
        weighted1 = tf.transform(p1)
        # transform using original method (row-wise)
        for n in range(p1.shape[0]):
            p1[n, :] = p1[n, :].multiply(weight2)
        weighted2 = p1

        self.assertTrue(np.allclose(weighted1.todense(), weighted2.todense()))
