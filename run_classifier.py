
import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

from tfkld import TFKLD
from utils import loguniform, sample_param_dists, combine_features, load_dir, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--weighting', default='tfkld')
    parser.add_argument('--dr_type', required=True)
    parser.add_argument('--n_components', type=int, default=400)
    parser.add_argument('--transductive', action='store_true')
    args = parser.parse_args()
    ngram_range = (1, 1)

    data = load_dir(args.path, include_dev=False)
    (labels_trn, p1_trn, p2_trn), (labels_tst, p1_tst, p2_tst) = data
    labels_trn, labels_tst = np.array(labels_trn), np.array(labels_tst)
    
    print("Vectorizing...")
    if args.weighting.lower() == 'tfkld':
        weighter = TFKLD(
            ngram_range=ngram_range# , stop_words='english'
        ).fit(p1_trn, p2_trn, labels_trn)
    elif args.weighting.lower() == 'tfidf':
        weighter = TfidfVectorizer(
            ngram_range=ngram_range# , stop_words='english'
        ).fit(p1_trn + p2_trn + p1_tst + p2_tst)
        
    p1_trn, p2_trn = weighter.transform(p1_trn), weighter.transform(p2_trn)
    p1_tst, p2_tst = weighter.transform(p1_tst), weighter.transform(p2_tst)

    print("Reducing...")
    dr = None
    if args.dr_type.lower() == 'nmf':
        dr = NMF(n_components=args.n_components)
    elif args.dr_type.lower() == 'svd':
        dr = TruncatedSVD(n_components=args.n_components, algorithm='arpack')
    else:
        raise ValueError("Unrecognized dr_type {}".format(args.dr_type))

    if args.transductive:
        reduced = sp.vstack([p1_trn, p2_trn, p1_tst, p2_tst])
        reduced = dr.fit_transform(reduced)
        p1_trn, p2_trn = np.split(reduced[:p1_trn.shape[0] * 2], 2)
        p1_tst, p2_tst = np.split(reduced[p1_trn.shape[0] * 2:], 2)
    else:
        reduced = sp.vstack([p1_trn, p2_trn])
        p1_trn, p2_trn = np.split(dr.fit_transform(reduced), 2)
        p1_tst, p2_tst = dr.transform(p1_tst), dr.transform(p2_tst)

    print("Training...")
    # parameters = sample_param_dists(
    #     [[('kernel', ['linear'], 1),
    #       ('C', loguniform(10, 0, 3), 6)],
    #      [('kernel', ['rbf'], 1),
    #       ('C', loguniform(10, 0, 3), 6),
    #       ('gamma', loguniform(10, -9, 3), 10)]])

    X_trn = normalize(combine_features(p1_trn, p2_trn))
    # clf = GridSearchCV(
    #     LinearSVC(penalty='l2', loss='hinge', dual=True, class_weight=None),
    #     [{'C': [0.1, 0.2, 0.5, 1, 10, 100]}], n_jobs=3, cv=5, verbose=1)
    clf = LinearSVC(penalty='l2', C=0.2, loss='hinge', dual=True, class_weight=None)
    clf.fit(X_trn, labels_trn)
    # report(clf.cv_results_)

    print("Testing...")
    y_pred = clf.predict(normalize(combine_features(p1_tst, p2_tst)))
    print(classification_report(labels_tst, y_pred))
    print(" accuracy: {:.3f}".format(accuracy_score(labels_tst, y_pred)))
