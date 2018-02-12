
import numpy as np
from scipy import sparse as sp
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

from tfkld import TFKLD
from dim_reduction import DimReduction
from utils import combine_features, load_dir, report


def f1_score(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    # Column sum - precision
    p = (1.0 * mat[0,0] / mat[:,0].sum()) + (1.0 * mat[1,1] / mat[:,1].sum())
    p = p / 2.0
    # Row sum - recall
    r = (1.0 * mat[0,0] / mat[0,:].sum()) + (1.0 * mat[1,1] / mat[1,:].sum())
    r = r / 2.0
    f1 = (2 * p * r) / (p + r)
    return f1, p, r


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--weighting', default='tfkld')
    parser.add_argument('--dr_method', required=True)
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
    dr = DimReduction(args.dr_method, args.n_components)
    if args.transductive:
        trn, tst = dr.fit_transform(sp.vstack([p1_trn, p2_trn]),
                                    sp.vstack([p1_tst, p2_tst]))
        (p1_trn, p2_trn), (p1_tst, p2_tst) = np.split(trn, 2), np.split(tst, 2)
    else:
        p1_trn, p2_trn = np.split(dr.fit_transform(sp.vstack([p1_trn, p2_trn])), 2)
        p1_tst, p2_tst = np.split(dr.transform(sp.vstack([p1_tst, p2_tst])), 2)

    print("Training...")
    # clf = GridSearchCV(
    #     LinearSVC(penalty='l2', loss='hinge', dual=True, class_weight=None),
    #     [{'C': [0.1, 0.2, 0.5, 1, 10, 100]}], n_jobs=3, cv=5, verbose=1)
    clf = LinearSVC(penalty='l2', C=0.2, loss='hinge', dual=True, class_weight=None)
    clf.fit(normalize(combine_features(p1_trn, p2_trn)), labels_trn)
    # report(clf.cv_results_)

    print("Testing...")
    y_pred = clf.predict(normalize(combine_features(p1_tst, p2_tst)))
    f1, p, r = f1_score(labels_tst, y_pred)
    acc = accuracy_score(labels_tst, y_pred)
    print(" * Accuracy: {:.3f}".format(acc))
    print(" * F1-score: {:.3f}".format(f1))
    print(" * Precision: {:.3f}".format(p))
    print(" * Recall {:.3f}".format(r))
