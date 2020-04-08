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
$ python create_test_train.py --data test_data.csv --text_column text --label_column labels --perc_test 0.3 --resample over
```

This should create two files called `train.csv` and `test.csv` with 30% of the data in the test set being in the test set and the data resample according the resample statement, which can be:
- `over` random oversampling to match the majority category
- `under` random undersampling to match the majority distribution
- leave out if you don't want to resample at all

The reminder of the arguments are:
- `--data` the data to split into a train and test dataset
- `--text_column` the column indicating text. Default is `text`
- ` --label_column` the columns indicating labels. Default is  `labels`
- `--perc_test` percantage of data which should be the  test set

---
## Classify using a single classifier
In general I recommend using the grid search, but one might want to test out specific classifiers so I made this the this case.

It is called from the terminal in the following way, but I do encourage the reader to check out the script as well
```
$ python classify.py --classifier nb
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
- `--ngrams` n-grams used, options include `unigram`, `bigram`, `trigram` and `4gram`. Default is `bigram`. *Note that is also uses lower levels n-gram as well, e.g. bigram also uses unigrams*. 
- `--min_wordcount` minimum wordcount required for a word to be included in the model. Default is `2`
- `--max_vocab` maximum desired vocabulary choosen among the most frequent words, which isn't removed by other conditions. Default is `None`, e.g. no maximum size is set
- `--max_perc_occurance` a word is removed it is appears in X percent of the document. Can be considered a corpus specific stopwordlist. Default is `0.5`
- `--clf_args` arguments to be passed to the classifier given as a dictionary. Default is `{}`, an empty dictionary

---
## Grid search
This is the meat of the scripts. Beware of not overdoing it, if you fit all the models at once this will take a long time.

The simplest use case is:
```
$ python grid_search.py --clfs nb ab
```
Which will print:
```
Calling grid search with the arguments:
        data: train.csv
        clfs: ['nb', 'ab']
        resampling: [None]
        grid_search_clf: True
        grid_seach_vectorization: True
        cv: 5
        text_column: text
        label_column: labels
Fitting 5 folds for each of 24 candidates, totalling 120 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:   38.1s
[Parallel(n_jobs=-1)]: Done 120 out of 120 | elapsed:  2.8min finished
Fitting 5 folds for each of 48 candidates, totalling 240 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:  2.0min
[Parallel(n_jobs=-1)]: Done 176 tasks      | elapsed: 14.3min
[Parallel(n_jobs=-1)]: Done 240 out of 240 | elapsed: 21.5min finished


The grid search is completed the results were:

Using the resampling method: None
        The best fit of the clf: nb, obtained a score of 0.9082, with the parameters:
                tfidf__use_idf = True
                vect__binary = True
                vect__lowercase = True
                vect__max_df = 0.5
                vect__min_df = 2
                vect__ngram_range = (1, 2)
        The best fit of the clf: ab, obtained a score of 0.9127, with the parameters:
                clf__base_estimator = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=2, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
                tfidf__use_idf = True
                vect__binary = True
                vect__lowercase = True
                vect__max_df = 0.5
                vect__min_df = 2
                vect__ngram_range = (1, 4)
```
This is a bit much, let's break it down. The first simply gives you of which argument have been given and how the function is being called. Note that if the number of fits is extremely high it is probably ideal to try with less testing. e.g. only using grid search for vectorization or for the classifiers or only using 3 cross validation folds.
The last bit shows the results of the grid search, which parameter were the best for each model. We will look into this in the next section.

You will see that this doesn't use the `test.csv` but you can test the best function using the `classify.py` above (see example in next section).

The possible arguments are:
- `--data` the desired data. Default is `train.csv`, i.e. can typically left out if the data is already split using `create_test_train.py`
- `--clfs` the classifiers to use, same options as `classify.py`. Can be multiple.
- `--grid_search_clf` should classifiers hyperparemeters be grid searched. Default is `True`
- `--grid_seach_vectorization` should vectorization hyperparemeters be grid searched. Vectorization is the method of turning the text into a bag-of-word and/or tf-idf. Default is `True`.
- `--cv` number of cross validation folds. Default is `5`
- `--resampling` same as `create_test_train.py` but can be a list. See to following code for an example. *Note that this is only relevant if the train test data isn't resampled*
- `--text_column` same as `create_test_train.py`
- `--label_column` same as `create_test_train.py`

The advanced use case is:
```
python grid_search.py --clfs nb en rf ab xg en --grid_search_clf True --grid_seach_vectorization True --cv 5
```
*Note* that here is used the train data and used the resampling. Naturally this assumed that the train data is not resampled. Basically this is a way to test the influence of the resampling and which approach is best. **Warning**: This will take a long time to run, but it should run through all the variations I have specified for each model combined with each method of vectorization. Maybe a bit too exhaustive as the method of vectorization is probably going to good for similar models.

Lastly, the variables used for the grid search is hidden with the script and are defined as:
- `search_params_vect` search parameter for the vectorization
- `search_params_tfidf` search parameter for the tfidf (e.g. whether it should or should use idf)
- `search_params_clf` these are the search parameters for the classifiers
You are free to change these in the script, however note that the grid search increase exponentially so I recommend starting adding thing slowly and removing things if things take too long. There is a lot of experimentation to do here, which is likely to influence results.


---
## Checking best classifier
This used the classify.py as mentioned above and the output from the first chunk of code for the grid search. We tested two model in the above code. Let's check how they perform on the test data. Starting with the nb the output were:
```
        The best fit of the clf: nb, obtained a score of 0.9082, with the parameters:
                tfidf__use_idf = True
                vect__binary = True
                vect__lowercase = True
                vect__max_df = 0.5
                vect__min_df = 2
                vect__ngram_range = (1, 2)
```
All of these as measure of the vectorization as the naive bayes classifier have no hyperparameter to optimize using grid search. This can also be seen the the prefix (`vect__*` and `tfidf__*`). These results mean that the best results were founds with the following setup:
- using tfidf (`tfidf__use_idf = True`)
- using a binary classifer (`vect__binary = True`)
- using lowercase (`vect__lowercase = True`)
- removing words which appeared in 50% of the documents (`vect__max_df = 0.5`) which is correspond to `max_perc_occurance` in the `classify.py` arguments
- removing words which only appear once in the training data. (`vect__min_df`) (Note that this is the training data in the cross validated dataset e.g. a subet of `train.csv`)
- the best performance was seen using bigrams

**Note** that not all options are checked. To see exactyl which ones are checked you will have to examine the script and it might be ideal to change these to explore different things.

The way to check this using `classify.py` would then be to:
```
python classify.py --classifier nb --max_perc_occurance 0.5 --binary True --lowercase True --use_tfidf True --ngrams bigram --min_wordcount 2
```

Let's also examine the case of the Adaboost:
```
        The best fit of the clf: ab, obtained a score of 0.9127, with the parameters:
                clf__base_estimator = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=2, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
                tfidf__use_idf = True
                vect__binary = True
                vect__lowercase = True
                vect__max_df = 0.5
                vect__min_df = 2
                vect__ngram_range = (1, 4)
```
You can see that the only argument tested for the classifier was base estimator (the only one with the `clf__*` prefix). Which is a bit long, but it essentially says that the best base estimator tested was a decision tree with a depth of 2 (the ones tested was decision trees with a depth of 1 and 2)

For the reminder we see that it performaned the best with 4 grams and otherwise the same as above. This could be written

```
python classify.py --classifier ab --max_perc_occurance 0.5 --binary True --lowercase True --use_tfidf True --ngrams 4gram --min_wordcount 2 --clf_args '{"base_estimator":"DecisionTreeClassifier(max_depth=2)"}'
```
The syntax for the argument clf_args is a bit exhaustive, but it is similar to json and dictionaries. Feel free to let me know if there is any issues.

---
***Final note***

You if you want to adjust more to the scripts you can create you own script and import the function. Such an example could e.g. look like:
```
from create_test_train import split_to_train_test

# now it is possible to call the imported function similar to how you do with any package:
split_to_train_test(...)
```