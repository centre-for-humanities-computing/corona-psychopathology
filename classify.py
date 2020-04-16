"""
a script to train and test classifier
"""
import argparse
import json

import pandas as pd
from utils import get_clf, resample

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier

def clf(train="train.csv",
        test="test.csv",
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
        clf_args={},
        scoring="accuracy",
        **kwargs):
    """
    Use for testing a single classifier
    """
    train = pd.read_csv(train)
    test = pd.read_csv(test)

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

    fit = pipe.fit(train[text_column], train[label_column])
    perf = {"acc_train": fit.score(train[text_column], train[label_column],
                                   scoring=scoring),
            "acc_test": fit.score(test[text_column], test[label_column],
                                  scoring=scoring)}
    print(f"Performance using the classifier {classifier}" +
          ", was found to be:")

    for k in perf:
        print(f"\t{k}: {round(perf[k], 4)}")
    return perf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--classifier",
                        help="what classifier to use",
                        default="nb")
    parser.add_argument("-ti", "--use_tfidf",
                        help="Should you use tfidf?",
                        default=True, type=bool)
    parser.add_argument("-l", "--lowercase",
                        help="Should you lowercase?",
                        default=True, type=bool)
    parser.add_argument("-b", "--binary",
                        help="Should you binarize tokens?",
                        default=False, type=bool)
    parser.add_argument("-ng", "--ngrams",
                        help="unigram, bigrams, or trigrams",
                        default="bigram")
    parser.add_argument("-mw", "--min_wordcount",
                        help="minimum wordcount",
                        default=2, type=int)
    parser.add_argument("-mv", "--max_vocab",
                        help="max_vocab",
                        default=None, type=int)
    parser.add_argument("-mp", "--max_perc_occurance",
                        help="max_perc_occurance",
                        default=0.5, type=float)
    parser.add_argument("-ca", "--clf_args",
                        help="argument given to the classifier",
                        default={}, type=json.loads)

    parser.add_argument("-tr", "--train",
                        help="trainset",
                        default="train.csv")
    parser.add_argument("-te", "--test",
                        help="testset",
                        default="test.csv")
    parser.add_argument("-tc", "--text_column",
                        help="columns for text",
                        default="text")
    parser.add_argument("-lc", "--label_column",
                        help="columns for labels",
                        default="labels")
    parser.add_argument("-sc", "--scoring",
                        help="scoring function for grid search",
                        default="accuracy")
    # parse and clean args:
    args = parser.parse_args()
    args = vars(args)  # make it into a dict

    ngram_d = {"unigram": (1, 1),
               "bigram": (1, 2),
               "trigram": (1, 3),
               "4gram": (1, 4)}
    if 'ngrams' in args:
        args['ngram_range'] = ngram_d[args['ngrams']]
        del args['ngrams']
    if ('clf_args' in args) and \
       ('base_estimator' in args['clf_args']):
        args['clf_args']['base_estimator'] = \
            eval(args['clf_args']['base_estimator'])

    clf(**args)
