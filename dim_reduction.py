
from sklearn.decomposition import NMF, TruncatedSVD
from scipy.sparse.linalg import svds


class DimReduction(object):
    def __init__(self, method, n_components):
        self.method = method
        self.n_components = n_components
        self.reducer = None

    def fit(self, m_trn, m_tst=None):
        if m_tst is not None:
            m = sp.vstack([m_trn, m_tst])
        else:
            m = m_trn

        if self.method.lower() == 'nmf':
            self.reducer = NMF(n_components=self.n_components).fit(m)
            
        elif self.method.lower() == 'svd':
            U, s, self.reducer = svds(m, k=self.n_components)

        else:
            raise ValueError("Unrecognized dr_type {}".format(self.method))

        return self

    def transform(self, m_trn, m_tst=None):
        if m_tst is not None:
            m = sp.vstack([m_trn, m_tst])
        else:
            m = m_trn

        if self.method.lower() == 'nmf':
            reduced = self.reducer.transform(m)
        else:
            reduced = m.dot(self.reducer.T)

        if m_tst is not None:
            return reduced[:m_trn.shape[0]], reduced[m_trn.shape[0]:]
        else:
            return reduced

    def fit_transform(self, m_trn, m_tst=None):
        self.fit(m_trn, m_tst=m_tst)
        return self.transform(m_trn, m_tst=m_tst)
