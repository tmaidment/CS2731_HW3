from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import numpy as np

class neural_spaCy(BaseEstimator, TransformerMixin):

    def __init__(self, *args, **kwargs):
        #self.path = get_tmpfile("word2vec.model")
        self.nlp = spacy.load('en_core_web_md')
        super().__init__()

    def transform(self, X, y=None):
        vectors = []
        for sentence in X:
            vectors.append(self.nlp(sentence).vector)
        return np.array(vectors)

    def fit(self, X, y=None):
        #self.model = Word2Vec(X, size=100, window=5, min_count=1, workers=4)
        #self.model.save("word2vec.model")
        return self