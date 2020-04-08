# Introduction
This is the github for the corona psychopathology

To download either download it via. github or go to the terminal and type to download it to the location in the terminal
```
git clone https://github.com/centre-for-humanities-computing/corona-psychopathology
```

Before you start you might neeed to create a virtual environment in anaconda or similar. If you have multiple version of python installed please make sure to use `python3` instead of `python` in the following commands.

---

## Creating the test and train set
To create the test and train set run the following, to run the script `create_test_train.py`
```
python create_test_train.py --data test_data.csv --text_column text --label_column labels --perc_test 0.3 --resample over
```

This should create two files called `train.csv` and `test.csv` with 30% of the data in the test set being in the test set and the data resample according the resample statement, which can be:
- `over` random oversampling to match the majority category
- `under` random undersampling to match the majority distribution
- leave out if you don't want to resample at all

---
## Classify using a single classifier
In general I recommend using the grid search, but one might want to test out specific classifiers so I made this the this case.

It is called from the terminal in the following way, but I do encourage the reader to check out the script as well
```
python classify.py --classifier nb
```
which should print out:
```
Performance using the classifier nb, was found to be:
        acc_train: 0.9594
        acc_test: 0.9312
```

Naturally you can choose many more options these include:
- `--classifier` the classifier to be used options include:
  - `nb`: naive bayes
  - `rf`: random forest
  - `ab`: Adaboost
  - `xg` XGboost, *Note* that this requires and external package which needs to be installed
  - `en` Elastic Net
- `--use_tfidf` should the preprocessing use tf-idf (alternative is Bag-of-words). Default is `True` 
- `--lowercase` should the text be lowercased. Default is `True`
- `--binary` should the text be binarized, e.g. only detect whether a word (or n-gram) appear rather than using count. Default is `False`
- `--ngrams` n-grams used, options include `unigram`, `bigram` and `trigram`. Default is `bigram`. *Note that is also uses lower levels n-gram as well, e.g. bigram also uses unigrams*. 
- `--min_wordcount` minimum wordcount required for a word to be included in the model. Default is `2`
- `--max_vocab` maximum desired vocabulary choosen among the most frequent words, which isn't removed by other conditions. Default is `None`, e.g. no maximum size is set
- `--max_perc_occurance` a word is removed it is appears in X percent of the document. Can be considered a corpus specific stopwordlist. Default is `0.5`
- `--clf_args` arguments to be passed to the classifier given as a dictionary. Default is `{}`, an empty dictionary




---
***Final note***

You if you want to adjust more to the scripts you can create you own script and import the function. Such an example could e.g. look like:
```
from create_test_train import split_to_train_test

# now it is possible to call the imported function:
split_to_train_test(...)
```