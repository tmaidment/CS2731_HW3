import numpy as np
from os import path
import re
from nltk.tokenize import word_tokenize

class dataset():
    def __init__(self, df, text_representation, X_label='comment_text', y_label='is_constructive', preprocessing=0):
        pre_processed_corpus = list(df[X_label])
        corpus = []
        assert preprocessing >= 0 and preprocessing <= 2
        for comment in pre_processed_corpus:
            # Different pre-processing experiments - negligable performance differences, but test them out!
            if preprocessing == 0:
                corpus.append(comment.lower())
            elif preprocessing == 1:
                corpus.append(re.sub('[^ .,a-zA-Z0-9]', '', comment.lower()))
            elif preprocessing == 2:
                corpus.append(' '.join(word_tokenize(re.sub('[^ .,a-zA-Z0-9]', '', comment.lower()))))
        vectorizer = text_representation
        X = vectorizer.fit_transform(corpus)
        y = list(df[y_label])
        if 'yes' in y:
            y = [item == 'yes' for item in y]
            y = np.array(y, dtype=int)
        else:
            y = [str(item).split('\n')[0] for item in y]
            y = np.array(y, dtype=float)
        self.X = X
        self.y = y
        self.titles = list(df['article_id'])

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_titles(self):
        return self.titles
    