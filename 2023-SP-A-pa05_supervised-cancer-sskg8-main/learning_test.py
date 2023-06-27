#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Autograder for pa04
"""

import os
import pandas as pd  # type: ignore
import supervised

if os.path.exists("train_grade.csv"):
    os.remove("train_grade.csv")
if os.path.exists("test_grade.csv"):
    os.remove("test_grade.csv")

# %%
# Grab a random 70% of all the data for train, and 30% for test
test_df = pd.read_csv("GSE73002_breast_cancer_test.csv")
train_df = pd.read_csv("GSE73002_breast_cancer_test.csv")
df_all = pd.concat([test_df, train_df])
train = df_all.sample(frac=0.7)
train.to_csv("train_grade.csv", index=False)
test = df_all.loc[~df_all.index.isin(train.index)]
test.to_csv("test_grade.csv", index=False)

my_learner = supervised.MyLearner()

X_train, y_train = supervised.import_data("train_grade.csv")
X_test, y_test = supervised.import_data("test_grade.csv")

my_learner.fit(data=X_train, target=y_train)
my_results = my_learner.predict(data=X_test)

your_grade = list(y_test == my_results).count(True) / len(y_test)
your_grade *= 70

with open("results.txt", mode="w") as fwrite:
    fwrite.write(str(round(your_grade)))
