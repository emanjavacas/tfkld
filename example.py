
import os

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from tfkld import TFKLD


def load_text(path, sep='\t', max_pairs=None):
    """
    Loads pairs from file in format: p1, p2, label
    """
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            if max_pairs is not None and idx >= max_pairs:
                break
            label, p1, p2 = line.strip().split(sep)
            try:
                yield int(label), p1, p2
            except ValueError:
                pass


def make_vectorizer(X_train, ngram_range=(1, 2)):
    """
    Auxiliary function to create a vectorizer
    """
    return CountVectorizer(ngram_range=ngram_range, dtype=np.float).fit(X_train)


def loguniform(base=10, low=0, high=1):
    """
    distribution for sampling according to a loguniform distribution
    """
    def rvs(size=None):
        return np.power(base, np.random.uniform(low, high, size))

    return rvs


def sample_param_dists(param_dists):
    sampled_params = []

    for param_dist in param_dists:

        sampled = {}
        for param, dist, n in param_dist:
            # assume uniform
            if isinstance(dist, list):
                sampled[param] = [np.random.choice(dist) for _ in range(n)]
            # assume sampler function
            elif callable(dist):
                sampled[param] = [dist() for _ in range(n)]
            else:
                raise ValueError(
                    '`dist` must be list or function but got {}'.format(type(dist)))

        sampled_params.append(sampled)

    return sampled_params


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--dr_type', required=True)
    parser.add_argument('--n_components', type=int, default=100)
    parser.add_argument('--transductive', action='store_true')
    args = parser.parse_args()

    train = os.path.join(args.path, 'train.data')
    test = os.path.join(args.path, 'test.data')
    dev = os.path.join(args.path, 'dev.data')
    labels_train, p1_train, p2_train = zip(*load_text(train, sep=args.sep))
    labels_test, p1_test, p2_test = zip(*load_text(test, sep=args.sep))
    labels_dev, p1_dev, p2_dev = zip(*load_text(dev, sep=args.sep))

    print("Vectorizing...")
    vec = make_vectorizer(p1_train + p2_train + p1_dev + p2_dev)
    p1_train, p2_train = vec.transform(p1_train), vec.transform(p2_train)
    p1_dev, p2_dev = vec.transform(p1_dev), vec.transform(p2_dev)
    p1_test, p2_test = vec.transform(p1_test), vec.transform(p2_test)

    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)
    labels_dev = np.array(labels_dev)

    print("Weighting...")
    tf = TFKLD().fit(
        sp.vstack([p1_train, p1_dev]),
        sp.vstack([p2_train, p2_dev]),
        np.concatenate([labels_train, labels_dev]))
    p1_train, p2_train = tf.transform(p1_train), tf.transform(p2_train)
    p1_dev, p2_dev = tf.transform(p1_dev), tf.transform(p2_dev)
    p1_test, p2_test = tf.transform(p1_test), tf.transform(p2_test)

    print("Reducing...")
    dr = None
    if args.dr_type.lower() == 'nmf':
        dr = NMF(n_components=args.n_components)
    elif args.dr_type.lower() == 'svd':
        dr = TruncatedSVD(n_components=args.n_components)
    else:
        raise ValueError("Unrecognized dr_type {}".format(args.dr_type))

    if args.transductive:
        X_train = sp.vstack([p1_train, p2_train, p1_dev, p2_dev, p1_test, p2_test])
        p1_train, p2_train, p1_dev, p2_dev, p1_test, p2_test = \
            np.split(dr.fit_transform(X_train), 6)

    else:
        X_train = sp.vstack([p1_train, p2_train, p1_dev, p2_dev])
        p1_train, p2_train, p1_dev, p2_dev = \
            np.split(dr.fit_transform(X_train), 4)
        p1_test, p2_test = dr.transform(p1_test), dr.transform(p2_test)

    print("Training...")
    parameters = sample_param_dists(
        [[('kernel', ['linear'], 1),
          ('C', loguniform(10, 0, 3), 6)],
         [('kernel', ['rbf'], 1),
          ('C', loguniform(10, 0, 3), 6),
          ('gamma', loguniform(10, -9, 3), 10)]])

    C = [1, 10, 100, 1000]
    parameters = [{'kernel': ['linear'], 'C': C},
                  {'kernel': ['rbf'], 'C': C, 'gamma': np.logspace(-9, 3, 13)}]

    clf = GridSearchCV(SVC(), parameters, n_jobs=3, cv=2)
    clf.fit(X_train, labels_train)
    report(clf.cv_results_)
