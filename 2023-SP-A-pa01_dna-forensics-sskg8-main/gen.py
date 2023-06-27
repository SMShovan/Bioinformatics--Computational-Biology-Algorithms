#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import skbio  # type: ignore

# import time


def random_sequence(moltype: skbio.Sequence, length: int) -> skbio.Sequence:
    result = []
    alphabet = list(moltype.nondegenerate_chars)
    for e in range(length):
        result.append(np.random.choice(alphabet))
    return moltype("".join(result))


def make_base_human() -> list[str]:
    base_human_conserved = []
    for segment in range(40):
        base_human_conserved.append(
            str(random_sequence(skbio.DNA, np.random.randint(10, 15)))
        )
    return base_human_conserved


def make_human(base_human_conserved: list[str]) -> list[str]:
    microsatellites = []
    for segment in range(40):
        microsatellites.append(
            str(random_sequence(skbio.DNA, np.random.randint(10, 15)))
        )
    return [
        item for pair in zip(base_human_conserved, microsatellites) for item in pair
    ]


def mate(
    base_human_conserved: list[str], parent1: list[str], parent2: list[str]
) -> list[str]:
    child_microsats = []
    # Think about which segments are being randomized!
    alt = 0
    for microsat_pos in range(1, len(parent1), 2):
        # Make percentages consistent for grading
        # if alt == 0:
        if np.random.randint(0, 2):
            child_microsats.append(parent1[microsat_pos])
            alt = 1
        else:
            child_microsats.append(parent2[microsat_pos])
            alt = 0
    return [
        item for pair in zip(base_human_conserved, child_microsats) for item in pair
    ]


base_human_conserved = make_base_human()

family = {}

# %25 similar genes to you in Microsat region
family["Maternal_Grandma_Grandparent"] = make_human(base_human_conserved)
family["Maternal_Grandpa_Grandparent"] = make_human(base_human_conserved)

family["Paternal_Grandma_Grandparent"] = make_human(base_human_conserved)
family["Paternal_Grandpa_Grandparent"] = make_human(base_human_conserved)

# %50 similar genes to you in Microsat region
family["Mother_Parent"] = mate(
    base_human_conserved,
    family["Maternal_Grandma_Grandparent"],
    family["Maternal_Grandpa_Grandparent"],
)
family["Father_Parent"] = mate(
    base_human_conserved,
    family["Paternal_Grandma_Grandparent"],
    family["Paternal_Grandpa_Grandparent"],
)

# %100
family["You"] = mate(
    base_human_conserved, family["Mother_Parent"], family["Father_Parent"]
)

# These write new random versions of your assignment files, in FASTA format
# Think about why I'm just taking the level (grandparent, parent).
# Can we reconstruct the difference betweet mother and father without more information?
keys = list(family.keys())
np.random.shuffle(keys)

with open("fam_unknown.fasta", "w") as f:
    for key, n in zip(keys, range(len(family.keys()))):
        if key == "You":
            f.write("> " + "Sequence_" + str(key.split("_")[-1]) + "\n")
        else:
            f.write("> " + "Sequence_" + str(n) + "\n")
        f.write("".join(family[key]) + "\n")

with open("fam_key.fasta", "w") as f:
    for key, n in zip(keys, range(len(family.keys()))):
        f.write("> " + "Sequence_" + str(key.split("_")[-1]) + "\n")
        f.write("".join(family[key]) + "\n")

# A sigle global alignment
# Don't run this unless you want to wait ~30 minutes
# t0 = time.time()
# aln = skbio.alignment.global_pairwise_align_nucleotide(
#     skbio.DNA("".join(family["Father"])), skbio.DNA("".join(family["Self_grandchild"]))
# )
# t1 = time.time()
# total = t1 - t0
# print("Runtime was: ", total / 60, " minutes")
