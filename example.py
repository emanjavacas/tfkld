
import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from tfkld import TFKLD
from utils import load_dir, make_vectorizer
from utils import loguniform, sample_param_dists
from utils import combine_features
from utils import report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--dr_type', required=True)
    parser.add_argument('--n_components', type=int, default=400)
    parser.add_argument('--transductive', action='store_true')
    args = parser.parse_args()

    data = load_dir(args.path, include_dev=False)
    (labels_train, p1_train, p2_train), (labels_test, p1_test, p2_test) = data

    print("Vectorizing...")
    vec = make_vectorizer(p1_train + p2_train)
    p1_train, p2_train = vec.transform(p1_train), vec.transform(p2_train)
    p1_test, p2_test = vec.transform(p1_test), vec.transform(p2_test)

    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    print("Weighting...")
    tf = TFKLD().fit(p1_train, p2_train, labels_train)
    p1_train, p2_train = tf.transform(p1_train), tf.transform(p2_train)
    p1_test, p2_test = tf.transform(p1_test), tf.transform(p2_test)

    print("Reducing...")
    dr = None
    if args.dr_type.lower() == 'nmf':
        dr = NMF(n_components=args.n_components)
    elif args.dr_type.lower() == 'svd':
        dr = TruncatedSVD(n_components=args.n_components, algorithm='arpack')
    else:
        raise ValueError("Unrecognized dr_type {}".format(args.dr_type))

    if args.transductive:
        reduced = sp.vstack([p1_train, p2_train, p1_test, p2_test])
        reduced = dr.fit_transform(reduced)
        p1_train, p2_train = np.split(reduced[:p1_train.shape[0] * 2], 2)
        p1_test, p2_test = np.split(reduced[p1_train.shape[0] * 2:], 2)
    else:
        reduced = sp.vstack([p1_train, p2_train])
        p1_train, p2_train = np.split(dr.fit_transform(reduced), 2)
        p1_test, p2_test = dr.transform(p1_test), dr.transform(p2_test)

    print("Training...")
    # parameters = sample_param_dists(
    #     [[('kernel', ['linear'], 1),
    #       ('C', loguniform(10, 0, 3), 6)],
    #      [('kernel', ['rbf'], 1),
    #       ('C', loguniform(10, 0, 3), 6),
    #       ('gamma', loguniform(10, -9, 3), 10)]])

    C = [1, 10, 100, 1000]
    parameters = [
        {'kernel': ['linear'], 'C': C},
        {'kernel': ['rbf'], 'C': C, 'gamma': np.logspace(-9, 3, 13)}]

    X_train = combine_features(p1_train, p2_train)
    clf = GridSearchCV(SVC(), parameters, n_jobs=3, cv=5, verbose=1)
    clf.fit(X_train, labels_train)
    report(clf.cv_results_)

    print("Testing...")
    X_test = combine_features(p1_test, p2_test)
    y_pred = clf.predict(X_test)
    print(classification_report(labels_test, y_pred))
