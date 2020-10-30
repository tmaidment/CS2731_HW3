from dataset import dataset#, save_dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from majorityvote import MajorityVote
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from latentsemanticanalysis import LatentSemanticAnalysis
from neural import neural_spaCy
from sklearn.model_selection import cross_validate
import numpy as np


def test_cv(dataset, classifier):
    X = dataset.get_X()
    y = dataset.get_y()
    titles = dataset.get_titles()
    cv_results = cross_validate(classifier, X, y, groups=titles, cv=5)
    ## NOTICE: MSE scores are sign-flipped, due to a design flaw
    ## (https://github.com/scikit-learn/scikit-learn/issues/2439)
    ## For that reason, I return the ABS value of each metric.
    average_score = np.abs(np.mean(cv_results['test_score']))
    fold_scores = [np.round(np.abs(result), decimals=4)  for result in cv_results['test_score']]
    average_time = np.abs(np.mean(cv_results['fit_time']) + np.mean(cv_results['score_time']))
    print('Average Score: {:.4f} | Fold Scores: {} | Average Calculation Time {:.4f}'.format(average_score, fold_scores, average_time))
    
def test_single(dataset, classifier):
    X = dataset.get_X()
    y = dataset.get_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    clf = classifier.fit(X, y)
    print('Score:', np.abs(clf.score(X_test, y_test)))

if __name__ == "__main__":
    print('=== TEXT CLASSIFICATION ===')
    print('\n--- BoW Representation Performance ---')
    bog_dataset = dataset(pd.read_excel('./SFUcorpus.xlsx'), CountVectorizer())
    test_cv(bog_dataset, MajorityVote())
    test_cv(bog_dataset, LogisticRegression(solver='lbfgs'))
    print('\n--- Sparse Representation Performance ---')
    sparse_dataset = dataset(pd.read_excel('./SFUcorpus.xlsx'), TfidfVectorizer())
    test_cv(sparse_dataset, MajorityVote())
    test_cv(sparse_dataset, LogisticRegression(solver='lbfgs'))
    print('\n--- Dense Representation Performance ---')
    dense_dataset = dataset(pd.read_excel('./SFUcorpus.xlsx'), LatentSemanticAnalysis())
    test_cv(dense_dataset, MajorityVote())
    test_cv(dense_dataset, LogisticRegression(solver='lbfgs'))
    print('\n--- Neural Representation Performance ---')
    neural_dataset = dataset(pd.read_excel('./SFUcorpus.xlsx'), neural_spaCy())
    test_cv(neural_dataset, MajorityVote())
    test_cv(neural_dataset, LogisticRegression(solver='lbfgs'))

    y_labels = ['is_constructive:confidence', 'toxicity_level:confidence']
    for y_label in y_labels:
        print('=== TEXT REGRESSION ===')
        print(f'Using the \'{y_label}\' column for regression values')
        print('\n--- BoW Representation Performance ---')
        bog_dataset = dataset(pd.read_excel('./SFUcorpus.xlsx'), CountVectorizer(), y_label=y_label)
        test_cv(bog_dataset, LinearRegression())
        test_cv(bog_dataset, svm.SVR(gamma='scale'))
        print('\n--- Sparse Representation Performance ---')
        sparse_dataset = dataset(pd.read_excel('./SFUcorpus.xlsx'), TfidfVectorizer(), y_label=y_label)
        test_cv(sparse_dataset, LinearRegression())
        test_cv(sparse_dataset, svm.SVR(gamma='scale'))
        print('\n--- Dense Representation Performance ---')
        dense_dataset = dataset(pd.read_excel('./SFUcorpus.xlsx'), LatentSemanticAnalysis(), y_label=y_label)
        test_cv(dense_dataset, LinearRegression())
        test_cv(dense_dataset, svm.SVR(gamma='scale'))
        print('\n--- Neural Representation Performance ---')
        neural_dataset = dataset(pd.read_excel('./SFUcorpus.xlsx'), neural_spaCy(), y_label=y_label)
        test_cv(neural_dataset, LinearRegression())
        test_cv(neural_dataset, svm.SVR(gamma='scale'))
    
    print('Done.')
