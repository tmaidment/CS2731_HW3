from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

class LatentSemanticAnalysis(BaseEstimator, TransformerMixin):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.vectorizer = CountVectorizer()

    def transform(self, X, y=None):
        X = self.vectorizer.fit_transform(X)
        svd = TruncatedSVD(n_components=300, n_iter=7, random_state=1)
        return svd.fit_transform(X)

    def fit(self, X, y=None):
        return self