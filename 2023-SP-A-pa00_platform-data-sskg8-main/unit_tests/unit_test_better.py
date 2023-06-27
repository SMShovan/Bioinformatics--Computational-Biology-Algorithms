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
from complement import complement
import random
import skbio

for _ in range(100):
    seq = "".join(random.choices("GCTA", k=random.randint(5, 50)))
    # Yea, this is kind of giving you one way to get the answer,
    # but I would prefer that you code it manually...
    assert complement(seq) == str(skbio.DNA(seq).complement())
