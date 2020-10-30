from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score

class MajorityVote(BaseEstimator, ClassifierMixin):  

    def __init__(self):
        super().__init__()


    def fit(self, X, y=None):
        assert y is not None
        self.prediction_ = float(stats.mode(y).mode[0])
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "prediction_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return np.array([self.prediction_ for x in X])