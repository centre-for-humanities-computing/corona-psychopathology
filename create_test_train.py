"""
A script for creating test train set
"""
import argparse

import pandas as pd

from sklearn.model_selection import train_test_split

from utils import resample as rs


def split_to_train_test(data="test_data.csv",
                        text_column="text",
                        label_column="labels",
                        resample="over",
                        perc_test=0.3,
                        sampling_strategy=1,
                        **kwargs):

    df = pd.read_csv(data)

    X_train, X_test, y_train, y_test = \
        train_test_split(df[text_column],
                         df[label_column],
                         test_size=perc_test)

    train = pd.DataFrame(zip(X_train, y_train),
                         columns=["text", "labels"])
    if resample:
        train = rs(train, "labels",
                   method=resample,
                   sampling_strategy=sampling_strategy,
                   **kwargs)

    test = pd.DataFrame(zip(X_test, y_test),
                        columns=["text", "labels"])
    train.to_csv("train.csv")
    test.to_csv('test.csv')
    return(train, test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data",
                        help="filename",
                        default="test_data.csv")
    parser.add_argument("-tc", "--text_column",
                        help="columns for text",
                        default="text")
    parser.add_argument("-lc", "--label_column",
                        help="columns for labels",
                        default="labels")
    parser.add_argument("-pt", "--perc_test",
                        help="percent data to use as train",
                        default=0.3, type=float)
    parser.add_argument("-rs", "--resample",
                        help="what method should you use to resample",
                        default=None)

    # parse and clean args:
    args = parser.parse_args()
    args = vars(args)  # make it into a dict

    print("Calling split_to_train_test with args:", args)
    split_to_train_test(**args)
