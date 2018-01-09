
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


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


def load_dir(path, sep='\t', max_pairs=None, include_dev=False):
    train = os.path.join(path, 'train.data')
    test = os.path.join(path, 'test.data')
    dev = os.path.join(path, 'dev.data')
    labels_train, p1_train, p2_train = zip(*load_text(train, sep=sep))
    labels_test, p1_test, p2_test = zip(*load_text(test, sep=sep))
    labels_dev, p1_dev, p2_dev = zip(*load_text(dev, sep=sep))

    if include_dev:
        return (labels_train, p1_train, p2_train), \
            (labels_test, p1_test, p2_test), \
            (labels_dev, p1_dev, p2_dev)

    else:
        # ignore dev split (we do CV instead)
        labels_train += labels_dev
        p1_train += p1_dev
        p2_train += p2_dev

        return (labels_train, p1_train, p2_train), \
            (labels_test, p1_test, p2_test)


def make_vectorizer(X_train, ngram_range=(1, 2)):
    """
    Auxiliary function to create a vectorizer
    """
    return CountVectorizer(ngram_range=ngram_range, dtype=np.float).fit(X_train)


def combine_features(p1, p2):
    """
    Combine sentence-pair features into single matrix
    """
    return np.hstack([p1 + p2, abs(p1 - p2)])


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
