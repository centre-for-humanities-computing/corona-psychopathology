"""
Script for transforming the trustpilot data into the right format
"""
import random

import pandas as pd

df = pd.read_csv("trustpilot-big.csv")

pos = df['text'][df.score == 5]
neg = df['text'][df.score == 1]

# create an unbalanced dataset
neg_s = random.sample(list(neg.values), 10_000)
pos_s = random.sample(list(neg.values), 1_000)

_ = [(i[0], 0) for i in neg_s] + [(i[0], 1) for i in pos_s]

df = pd.DataFrame(_, columns = ['text', 'labels'])
df.to_csv("test_data.csv")
