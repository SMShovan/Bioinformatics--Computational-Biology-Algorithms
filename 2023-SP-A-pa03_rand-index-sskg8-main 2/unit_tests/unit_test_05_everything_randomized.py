#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your code tests start here:
# To debug in pudb3
# Highlight the line of code below below
# Type 't' to jump 'to' it
# Type 's' to 'step' deeper
# Type 'n' to 'next' over
# Type 'f' or 'r' to finish/return a function call and go back to caller
import numpy as np
from rand import contingency_table
from rand import confusion_table
from rand import rand
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.metrics.cluster import rand_score

for _ in range(10):
    n_categories_true = np.random.randint(4, 8)
    n_categories_pred = np.random.randint(4, 16)
    n_data = np.random.randint(32, 64)
    actuald = np.random.randint(0, n_categories_true, (n_data,))
    guesses = np.random.randint(0, n_categories_pred, (n_data,))

    contingency = contingency_table(actuald, guesses)
    assert (contingency_matrix(actuald, guesses) == contingency).all()
    confusion = confusion_table(contingency)
    assert (pair_confusion_matrix(actuald, guesses) == confusion).all()
    assert abs(rand_score(actuald, guesses) - rand(confusion)) < 0.001
