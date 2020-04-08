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

##


---
***Final note***

You if you want to adjust more to the scripts you can create you own script and import the function. Such an example could e.g. look like:
```
from create_test_train import split_to_train_test

# now it is possible to call the imported function:
split_to_train_test(...)
```