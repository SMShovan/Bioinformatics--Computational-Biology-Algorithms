#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import skbio  # type: ignore
import matplotlib.pyplot as plt
import functools
import skbio
import scipy as sp
from scipy import cluster
from typing import Callable

from IPython.core import page

global_pairwise_align_nucleotide = functools.partial(
    skbio.alignment.global_pairwise_align_nucleotide, penalize_terminal_gaps=True
)

def pairwise_alignment(seq1, seq2, gap_penalty=-1, match_score=2, mismatch_score=-1): # type: ignore
    """Performs pairwise sequence alignment using the Needleman-Wunsch algorithm.

    Args:
        seq1 (str): The first sequence to be aligned.
        seq2 (str): The second sequence to be aligned.
        gap_penalty (int, optional): The gap penalty (default: -1).
        match_score (int, optional): The score for a matching pair (default: 2).
        mismatch_score (int, optional): The score for a mismatched pair (default: -1).

    Returns:
        Tuple of aligned sequences and the alignment score.

    """
    # Initialize the scoring matrix
    n, m = len(seq1), len(seq2)
    score_matrix = np.zeros((n+1, m+1))
    score_matrix[:, 0] = np.arange(n+1) * gap_penalty
    score_matrix[0, :] = np.arange(m+1) * gap_penalty

    # Fill in the scoring matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
            delete = score_matrix[i-1][j] + gap_penalty
            insert = score_matrix[i][j-1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)

    # Traceback to find the optimal alignment
    align1, align2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        score = score_matrix[i][j]
        score_diag = score_matrix[i-1][j-1]
        score_up = score_matrix[i][j-1]
        score_left = score_matrix[i-1][j]

        if score == score_diag + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif score == score_left + gap_penalty:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        elif score == score_up + gap_penalty:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1

    # Add any remaining characters from the longer sequence
    while i > 0:
        align1.append(seq1[i-1])
        align2.append('-')
        i -= 1
    while j > 0:
        align1.append('-')
        align2.append(seq2[j-1])
        j -= 1

    # Reverse the alignments
    align1 = ''.join(reversed(align1))
    align2 = ''.join(reversed(align2))

    return align1, align2 , score_matrix[n][m]

def kmer_distance(sequence1, sequence2, k=10, overlap=True): # type: ignore
    """Compute the kmer distance between a pair of sequences

    Parameters
    ----------
    sequence1 : skbio.Sequence
    sequence2 : skbio.Sequence
    k : int, optional
        The word length.
    overlapping : bool, optional
        Defines whether the k-words should be overlapping or not
        overlapping.

    Returns
    -------
    float
        Fraction of the set of k-mers from both sequence1 and
        sequence2 that are unique to either sequence1 or
        sequence2.

    Raises
    ------
    ValueError
        If k < 1.

    Notes
    -----
    k-mer counts are not incorporated in this distance metric.

    """
    sequence1_kmers = set(map(str, sequence1.iter_kmers(k=k, overlap=overlap)))
    sequence2_kmers = set(map(str, sequence2.iter_kmers(k=k, overlap=overlap)))
    all_kmers = sequence1_kmers | sequence2_kmers
    shared_kmers = sequence1_kmers & sequence2_kmers
    number_unique = len(all_kmers) - len(shared_kmers)
    fraction_unique = number_unique / len(all_kmers)
    return fraction_unique

def guide_tree_from_sequences(
    sequences: list[skbio.Sequence],
    metric: Callable[
        [skbio.Sequence, skbio.Sequence, int, bool], float
    ] = kmer_distance,
    display_tree: bool = False,
) -> sp.cluster.hierarchy.ClusterNode: # type: ignore
    """Build a UPGMA tree by applying metric to sequences

    Parameters
    ----------
    sequences : list of skbio.Sequence objects (or subclasses)
      The sequences to be represented in the resulting guide tree.
    metric : function
      Function that returns a single distance value when given a pair of
      skbio.Sequence objects.
    display_tree : bool, optional
      Print the tree before returning.

    Returns
    -------
    skbio.TreeNode

    """
    guide_dm = skbio.DistanceMatrix.from_iterable(
        iterable=sequences, metric=metric, key="id"
    )
    guide_lm = sp.cluster.hierarchy.average(y=guide_dm.condensed_form())
    guide_tree = sp.cluster.hierarchy.to_tree(Z=guide_lm)
    if display_tree:
        guide_d = sp.cluster.hierarchy.dendrogram(
            Z=guide_lm,
            labels=guide_dm.ids,
            orientation="right",
            link_color_func=lambda x: "black",
        )
    return guide_tree

def progressive_msa(sequences, pairwise_aligner, guide_tree=None, metric=kmer_distance): # type: ignore
    """Perform progressive msa of sequences

    Parameters
    ----------
    sequences : skbio.SequenceCollection
        The sequences to be aligned.
    metric : function, optional
      Function that returns a single distance value when given a pair of
      skbio.Sequence objects. This will be used to build a guide tree if one
      is not provided.
    guide_tree : skbio.TreeNode, optional
        The tree that should be used to guide the alignment process.
    pairwise_aligner : function
        Function that should be used to perform the pairwise alignments,
        for example skbio.alignment.global_pairwise_align_nucleotide. Must
        support skbio.Sequence objects or skbio.TabularMSA objects
        as input.

    Returns
    -------
    skbio.TabularMSA

    """

    if guide_tree is None:
        guide_dm = skbio.DistanceMatrix.from_iterable(
            iterable=sequences, metric=metric, key="id"
        )
        guide_lm = sp.cluster.hierarchy.average(y=guide_dm.condensed_form())
        guide_tree = skbio.TreeNode.from_linkage_matrix(
            linkage_matrix=guide_lm, id_list=guide_dm.ids
        )

    seq_lookup = {s.metadata["id"]: s for i, s in enumerate(sequences)}

    # working our way down, first children may be super-nodes,
    # then eventually, they'll be leaves
    c1, c2 = guide_tree.children

    # Recursive base case
    if c1.is_tip():
        c1_aln = seq_lookup[c1.name]
    else:
        c1_aln = progressive_msa(
            sequences=sequences, pairwise_aligner=pairwise_aligner, guide_tree=c1
        )

    if c2.is_tip():
        c2_aln = seq_lookup[c2.name]
    else:
        c2_aln = progressive_msa(
            sequences=sequences, pairwise_aligner=pairwise_aligner, guide_tree=c2
        )

    # working our way up, doing alignments, from the bottom up
    alignment, _, _ = pairwise_aligner(seq1=c1_aln, seq2=c2_aln)

    # this is a temporary hack as the aligners in skbio 0.4.1 are dropping
    # metadata - this makes sure that the right metadata is associated with
    # the sequence after alignment
    if isinstance(c1_aln, skbio.Sequence):
        alignment[0].metadata = c1_aln.metadata
        len_c1_aln = 1
    else:
        for i in range(len(c1_aln)):
            alignment[i].metadata = c1_aln[i].metadata
        len_c1_aln = len(c1_aln)
    if isinstance(c2_aln, skbio.Sequence):
        alignment[1].metadata = c2_aln.metadata
    else:
        for i in range(len(c2_aln)):
            alignment[len_c1_aln + i].metadata = c2_aln[i].metadata

    # feed alignment back up, for further aligment, or eventually final return
    return alignment


def hamming_distance(seq1, seq2): # type: ignore
    """Calculate the Hamming distance between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length.")

    # Calculate the number of mismatches between the sequences
    mismatches = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

    return mismatches

def progressive_msa_and_tree(
    sequences,
    pairwise_aligner,
    metric=kmer_distance,
    guide_tree=None,
    display_aln=False,
    display_tree=False,
): # type: ignore
    """Perform progressive msa of sequences and build a UPGMA tree
    Parameters
    ----------
    sequences : skbio.SequenceCollection
        The sequences to be aligned.
    pairwise_aligner : function
        Function that should be used to perform the pairwise alignments,
        for example skbio.alignment.global_pairwise_align_nucleotide. Must
        support skbio.Sequence objects or skbio.TabularMSA objects
        as input.
    metric : function, optional
      Function that returns a single distance value when given a pair of
      skbio.Sequence objects. This will be used to build a guide tree if one
      is not provided.
    guide_tree : skbio.TreeNode, optional
        The tree that should be used to guide the alignment process.
    display_aln : bool, optional
        Print the alignment before returning.
    display_tree : bool, optional
        Print the tree before returning.

    Returns
    -------
    skbio.alignment
    skbio.TreeNode

    """
    msa = progressive_msa(
        sequences=sequences, pairwise_aligner=pairwise_aligner, guide_tree=guide_tree
    )

    if display_aln:
        print(msa)

    msa_dm = skbio.DistanceMatrix.from_iterable(iterable=msa, metric=metric, key="id")
    msa_lm = sp.cluster.hierarchy.average(y=msa_dm.condensed_form())
    msa_tree = skbio.TreeNode.from_linkage_matrix(
        linkage_matrix=msa_lm, id_list=msa_dm.ids
    )
    if display_tree:
        print("\nOutput tree:")
        d = sp.cluster.hierarchy.dendrogram(
            msa_lm,
            labels=msa_dm.ids,
            orientation="right",
            link_color_func=lambda x: "black",
        )
    return msa, msa_tree

def get_ranks(lst): # type: ignore
    """
    Given an unordered list of numbers, returns a list of their ranks.
    """
    sorted_lst = sorted(lst)
    ranks = []
    for num in lst:
        rank = sorted_lst.index(num) + 1
        ranks.append(rank)
    return ranks

# main function 

# Input start
sequences = []
keys = []

with open("fam_unknown.fasta", "r") as file:
    for i, line in enumerate(file):
        if i % 2 != 0:
            sequences.append(line.strip())
        else:
            keys.append(line.strip())
    
query_sequences = []

for i in range(len(sequences)):
    query_sequences.append(skbio.DNA(sequence=str(sequences[i]), metadata={"id": keys[i]}))

# Input end

# Guide Tree start
guide_dm = skbio.DistanceMatrix.from_iterable(
    iterable=query_sequences, metric=kmer_distance, key="id"
)

guide_lm = sp.cluster.hierarchy.average(y=guide_dm.condensed_form())
guide_tree = skbio.TreeNode.from_linkage_matrix(
    linkage_matrix=guide_lm, id_list=guide_dm.ids
)

# Guide Tree end



# Progressive MSA Start

# msa = progressive_msa(
#     sequences=query_sequences,
#     pairwise_aligner=global_pairwise_align_nucleotide,
#     guide_tree=guide_tree,
# )
# print(msa)

# Progressive MSA End


msa, tree = progressive_msa_and_tree(
    sequences=query_sequences,
    pairwise_aligner=global_pairwise_align_nucleotide,
    display_tree=False,
    display_aln=False,
)

my = -1
for i in range(len(keys)):
    if keys[i].split("_")[-1] == 'You':
        my = i
        break
my_output = -1
for i in range(len(sequences)):
    if sequences[my] == str(msa[i]).replace('-',""):
        my_output = i
        break

# print(sequences[my]+ "\n")
# print(str(msa[my_output]).replace('-',""))

hamming_distances = []

for i in range (len(sequences)):
    hamming_distances.append(hamming_distance(str(msa[my_output]) , str(msa[i])))
ranks = get_ranks(hamming_distances)

result = []

for i in range (len(sequences)):
    for j in range (len(sequences)):
        if len(str(sequences[i])) == len(str(msa[j]).replace('-',"")):
            if hamming_distance(str(sequences[i]) , str(msa[j]).replace('-',"")) == 0:
                

                if (ranks[j] == 1):
                    label = 'You'
                if (ranks[j] == 2):
                    label = 'Parent'
                if (ranks[j] == 3):
                    label = 'Parent'
                if (ranks[j] == 4):
                    label = 'Grandparent'
                if (ranks[j] == 5):
                    label = 'Grandparent'
                if (ranks[j] == 6):
                    label = 'Grandparent'
                if (ranks[j] == 7):
                    label = 'Grandparent'
                
                result.append("> Sequence_"+ str(label))
                result.append(str(str(msa[j]).replace('-',"")))

                print("> Sequence_"+ str(label))
                print(str(str(msa[j]).replace('-',"")))

                break

