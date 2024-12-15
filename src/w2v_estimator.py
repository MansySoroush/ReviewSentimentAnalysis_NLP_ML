
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
import numpy as np

# Define a custom wrapper for Word2Vec to use with scikit-learn
class Word2VecEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, sg=0, min_count=1, epochs=10, alpha=0.025):
        self.vector_size = vector_size
        self.window = window
        self.sg = sg
        self.min_count = min_count
        self.epochs = epochs
        self.alpha = alpha
        self.model = None

    def fit(self, sentences, y=None):
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            sg=self.sg,
            min_count=self.min_count,
            epochs=self.epochs,
            alpha=self.alpha,
        )
        return self

    def transform(self, sentences):
        # Average Word2Vec vectors for each sentence
        return np.array([
            np.mean([self.model.wv[word] for word in sentence if word in self.model.wv] or [np.zeros(self.vector_size)], axis=0)
            for sentence in sentences
        ])
