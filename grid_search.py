"""
A script for conducting grid search using sklearn
"""
import argparse

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from utils import get_clf, resample


def grid_search(data="train.csv",
                text_column="text",
                label_column="labels",
                clfs=['nb', 'rf', 'en', 'ab', 'xg'],
                resampling=[None],
                grid_search_clf=True,
                grid_seach_vectorization=True,
                cv=5,
                **kwargs
                ):
    """
    """
    df = pd.read_csv(data)

    # These are the variables checked for the grid search for vectorization
    # feel free to add to add or remove from this
    search_params_vect = {'ngram_range': [(1, 2), (1, 3), (1, 4)],
                          'lowercase': [True],
                          'max_df': [0.5],
                          'min_df': [2, 5],
                          'binary': [True, False],
                          }
    search_params_tfidf = {'use_idf': (True, False)}

    # These are the variables used for the grid search for classfiers
    # feel free to add or remove from this
    search_params_clf = \
        {'nb': {},
         'rf': {'max_depth': [None, 5, 10],
                'n_estimators': [50, 100, 200, 300],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]},
         'en': {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "l1_ratio": np.arange(0.0, 1.0, 0.1)},
         'ab': {'base_estimator': [DecisionTreeClassifier(max_depth=1),
                                   DecisionTreeClassifier(max_depth=2)]},
         'xg': {'learning_rate': [0.1, 0.2, 0.3],
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 12, 18],
                'min_child_weight': [1, 2, 3],
                'gamma': [0.0, 0.1, 0.2]}}

    # Defining an inner function for grid search
    def __grid_search(df, c, **kwargs):
        clf = get_clf(c, **kwargs)
        pipe = Pipeline([('vect', CountVectorizer(**kwargs)),
                        ('tfidf', TfidfTransformer(**kwargs)),
                        ('clf', clf(**kwargs))])

        # construct grid search parameters
        parameters = {}

        if grid_seach_vectorization:
            for k in search_params_vect:
                parameters['vect__'+k] = search_params_vect[k]
            for k in search_params_tfidf:
                parameters['tfidf__'+k] = search_params_tfidf[k]
        if grid_search_clf:
            for k in search_params_clf[c]:
                parameters['clf__'+k] = search_params_clf[c][k]

        gs_clf = GridSearchCV(pipe, parameters, cv=cv, verbose=True,
                              n_jobs=-1)  # run on all cores
        fit = gs_clf.fit(df[text_column], df[label_column])
        return(fit.best_score_, fit.best_params_, fit)

    results = {}
    for rs in resampling:
        res = {}
        df_ = resample(df, label_column, rs, **kwargs)
        for c in clfs:
            r = __grid_search(df_, c, **kwargs)
            res[c] = r
        results[rs] = res

    print("\n\nThe grid search is completed the results were:")
    for rs in results:
        print(f"\nUsing the resampling method: {rs}")
        for c in results[rs]:
            score, best_params, t = results[rs][c]
            print(f"\tThe best fit of the clf: {c}, " +
                  "obtained a score of {score}, with the parameters:")
            for p in best_params:
                print(f"\t\t{p} = {best_params[p]}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data",
                        help="what data to use",
                        default="train.csv")
    parser.add_argument("-c", "--clfs",
                        help="What classifier should you use",
                        nargs='+')
    parser.add_argument("-rs", "--resampling",
                        help="should you resample, and how. Can be multiple",
                        nargs='+', default=[None])
    parser.add_argument("-gc", "--grid_search_clf",
                        help="should you grid search classifier?",
                        default=True, type=bool)
    parser.add_argument("-gv", "--grid_seach_vectorization",
                        help="should you grid search vectorizer?",
                        default=True, type=bool)
    parser.add_argument("-cv", "--cv",
                        help="number of cross validation folds",
                        default=5, type=int)

    parser.add_argument("-tc", "--text_column",
                        help="columns for text",
                        default="text")
    parser.add_argument("-lc", "--label_column",
                        help="columns for labels",
                        default="labels")

    # parse and clean args:
    args = parser.parse_args()
    args = vars(args)  # make it into a dict

    print("\n\nCalling grid search with the arguments:")
    for k in args:
        print(f"\t{k}: {args[k]}")
    grid_search(**args)
