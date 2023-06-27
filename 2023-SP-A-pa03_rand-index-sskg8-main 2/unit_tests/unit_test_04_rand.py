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
from rand import rand
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.metrics.cluster import rand_score

actuald = np.array(["a", "a", "b", "c"])
guesses = np.array([0, 1, 1, 2])

confusion = pair_confusion_matrix(actuald, guesses)
assert abs(rand_score(actuald, guesses) - rand(confusion)) < 0.1
