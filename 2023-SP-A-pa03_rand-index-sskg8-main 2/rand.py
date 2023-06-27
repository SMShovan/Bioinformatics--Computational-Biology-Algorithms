#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
# Remember: don't import or use anything else besides base python and numpy!


def n_choose_2(n: int) -> int:
    return n * (n - 1) // 2


def contingency_table(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:

    return np.array([[np.sum((labels_true == u) & (labels_pred == v))
    for v in np.unique(labels_pred)]
    for u in np.unique(labels_true)])


def confusion_table(contingency: np.ndarray) -> np.ndarray:


    tp = np.sum([n_choose_2(n) for n in contingency.flatten() if n > 1])
    fn = np.sum([n_choose_2(n) for n in np.sum(contingency, axis=1)]) - tp
    fp = np.sum([n_choose_2(n) for n in np.sum(contingency, axis=0)]) - tp
    tn = n_choose_2(contingency.sum()) - tp - fn - fp
    return np.array([[tn*2, fp*2], [fn*2, tp*2]])
   
    


def rand(confusion: np.ndarray) -> float:    

    tp, fp, fn, tn = confusion.flatten()
    return (tp + tn) / (tp + fp + fn + tn)


if __name__ == "__main__":
    # This is a demonstration of how your functions could be used.
    # They should match the ones from sklearn and scipy,
    # but not use those as imports!
    from scipy.special import comb
    from sklearn.metrics.cluster import contingency_matrix
    from sklearn.metrics.cluster import pair_confusion_matrix
    from sklearn.metrics.cluster import rand_score

    labels_true = np.array(
        [
            "x",
            "x",
            "x",
            "x",
            "x",
            "o",
            "x",
            "o",
            "o",
            "o",
            "o",
            "v",
            "x",
            "x",
            "v",
            "v",
            "v",
        ]
    )
    labels_pred = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

    print("Your n_choose_2 with n=7\n", n_choose_2(7))
    print("SciPy's comb with n=7, k=2\n", comb(7, 2))

    contingency = contingency_table(labels_true, labels_pred)
    print("\nYour contingency:\n", contingency)
    print("sklearn's contingency:\n", contingency_matrix(labels_true, labels_pred))

    confusion = confusion_table(contingency)
    print("\nYour confusion:\n", confusion)
    print("sklearn's confusion:\n", pair_confusion_matrix(labels_true, labels_pred))

    print("\nYour rand:", rand(confusion))
    print("sklearn's rand:", rand_score(labels_true, labels_pred))
