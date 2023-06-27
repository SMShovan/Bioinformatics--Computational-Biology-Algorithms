#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys


def complement(num: str) -> str:
    """
    Purpose:        Converts a DNA sequence to its complement
    Parameters:     Sequence of DNA as a python str
    User Input:     No
    Prints:         Nothing
    Returns:        Result new string of DNA
    Modifies:       Nothing
    Calls:          Basic python only
    Tests:          ./unit_tests/*
    Status:         Done!

    Usage illustrated via some simple doctests:
    >>> complement("GATTACA")
    'CTAATGT'

    >>> complement("CAT")
    'GTA'

    >>> print("Unlike other frameworks, doctest does stdout easily")
    Unlike other frameworks, doctest does stdout easily
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    pass
    # YOUR CODE GOES HERE

    comp = {" ": " ", "A": "T", "C": "G", "G": "C", "T": "A", "\n": ""}

    complement_sequence = []

    for i in num:
        # print(complement[i])
        complement_sequence.append(comp[i])
        result = "".join(complement_sequence)

    return result


if __name__ == "__main__":
    # Execute doctests to protect main:
    import doctest

    doctest.testmod()

    # Run main:
    pass
    # YOUR CODE GOES HERE
    if len(sys.argv) == 3:
        temp = complement(open(sys.argv[1], "r").read())
        f = open(sys.argv[2], "w+")
        f.write(temp + "\n")
    else:
        print(complement(input("")))
