"""
A pipeline Danish text classification using scipy
"""
import argparse

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def get_clf(clf, calibrate=False, calibrate_cv=5, **kwargs):
    clf_dict = {"nb": MultinomialNB,
                "rf": RandomForestClassifier,
                "en": ElasticNet,
                'ab': AdaBoostClassifier
                }
    if clf == "xg":
        from xgboost import XGBClassifier
        clf_dict["xg"] = XGBClassifier
    if clf in clf_dict:
        clf = clf_dict[clf]
    else:
        raise ValueError(f"classifier {clf} is not a valid classifier")
    if calibrate:
        clf = CalibratedClassifierCV(clf, cv=cv)
    return clf


def resample(df, label_column, method="over", sampling_strategy=1, **kwargs):
    """
    method ('over'|'under'):
    """
    if method == "under":
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
        df, y = rus.fit_resample(df, df[label_column])
    elif method == "over":
        ros = RandomOverSampler(sampling_strategy=sampling_strategy)
        df, y = ros.fit_resample(df, df[label_column])
    else:
        raise ValueError("Invalid method {method}")
    return df


def single_clf(data="test_data.csv",
               text_column="text",
               label_column="labels",
               use_tfidf=True,
               lowercase=True,
               binary=False,
               ngram_range=(1, 2),
               min_wordcount=2,
               max_vocab=None,
               max_perc_occurance=0.5,
               classifier="nb",
               calibrate_clf=False,
               calibrate_cv=5,
               resample_method='over',
               cv=5,
               perc_test=0.3,
               **kwargs):
    """
    Use for testing a single classifier

    Example:
    >>> single_clf()
    """
    df = pd.read_csv(data)

    # create bag of word representations
    vectorizer = CountVectorizer(  # should text be lowercased
                                 lowercase=lowercase,
                                 # regular expression denoting a token
                                 token_pattern=r'(?u)\b\w\w+\b',
                                 # (1, 1) is unigram, (1, 2) unigram and
                                 # bigrams, (2, 2) is only bigrams and so on.
                                 ngram_range=ngram_range,
                                 # stopword list to use (given as a list)
                                 stop_words=None,
                                 # ignore word which appear in a given
                                 # percentage of the documents - e.g. 0.7 means
                                 # ignore words which appear in 70% percent of
                                 # the docs
                                 max_df=max_perc_occurance,
                                 # ignore wordcounts strictly lower than
                                 min_df=min_wordcount,
                                 # maximum number of words to use
                                 max_features=max_vocab,
                                 # only detect whether a word is there or not?
                                 binary=binary)

    clf = get_clf(classifier, calibrate_clf, calibrate_cv)

    if use_tfidf:
        pipe = Pipeline([('vect', vectorizer),
                        ('tfidf', TfidfTransformer()),
                        ('clf', clf())])
    else:
        pipe = Pipeline([('vect', vectorizer),
                        ('clf', clf(**kwargs))])

    if resample_method:
        print(f"Resampling dataset using the method {resample_method}")
        df = resample(df, label_column, resample_method, **kwargs)

    if cv:
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        perf = cross_validate(pipe, X=df[text_column], y=df[label_column],
                              cv=5,
                              return_train_score=True,
                              scoring=scoring,
                              verbose=True,
                              n_jobs=-1)  # run on all possible cores
        print(f"Performance using the classifier {classifier} (using cv), " +
              "was found to be:")
    else:
        X_train, X_test, y_train, y_test = \
            train_test_split(df[text_column],
                             df[label_column],
                             test_size=perc_test)
        fit = pipe.fit(X_train, y_train)
        perf = {"acc_train": fit.score(X_train, y_train),
                "acc_test": fit.score(X_test, y_test)}
        print(f"Performance using the classifier {classifier} (not using cv)" +
              ", was found to be:")

    for k in perf:
        print(f"\t{k}: {perf[k]}")
    return perf


def grid_search(data="test_data.csv",
                text_column="text",
                label_column="labels",
                clfs=['nb', 'rf', 'en', 'ab', 'xg'],
                resampling=[None, 'over', 'under'],
                grid_search_clf=True,
                grid_seach_vectorization=True,
                cv=5,
                perc_test=0.3,
                **kwargs
                ):
    """
    """
    df = pd.read_csv(data)
    X_train, X_test, y_train, y_test = train_test_split(df[text_column],
                                                        df[label_column],
                                                        test_size=perc_test)

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


        n = len(ParameterGrid(parameters))*cv
        if n > 1000:
            print(f"Number of fits is above 1000 ({n} exactly) " +
                  "This might take a long time to run." +
                  " Are you sure that is what you want?" +
                  " Consider using fewer using a subset og resampling" +
                  ", clfs or maybe lowering the number of cv's.")
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
                        help="filename",
                        default="test_data.csv")
    parser.add_argument("-l", "--data",
                        help="filename",
                        default="test_data.csv")
    # bow or tfidf
    parser.add_argument("-v", "--vectorizer",
                        help="What vectorizer should be used",
                        default="bow")
    parser.add_argument("-sw", "--stopwords",
                        help="a .txt files of Stopwords")
    parser.add_argument("-l", "--lowercase",
                        help="lowercase",
                        default="True")
    args = parser.parse_args()
    args = vars(args)  # make it into a dict

    if args["lowercase"].lower() == "true":
        args["lowercase"] = True
    elif args["lowercase"].lower() == "false":
        args["lowercase"] = False
    else:
        raise ValueError("Invalid value for lowercase, it must be either true \
                          or false")

    print("\n\nRunning pipeline with the arguments:")
    for k in args:
        print(f"\t{k}={args[k]}")

    main(**args)

