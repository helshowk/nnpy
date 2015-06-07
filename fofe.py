#!/usr/bin/env python2


import numpy as np


def vector_sequence(word_sequence, word_dict):
    """
    Build a vector sequence out of vectors in word_dict.
    """
    return [ word_dict[w] for w in word_sequence ]

def one_hot(word, word_seq):
    # get the one hot encoding vector of a word in a word list
    
    word_list = word_seq
    v = np.zeros(len(word_list))
    idx = word_list.index(word)
    v[idx] = 1
    return v

def biFOFE(sequence, alpha=0.7):
    """
    Create bigram FOFE encoding for a sequence given a forgetting factor alpha.  Note there is no lookup step here, sequence is assumed to accept * operator.

    http://arxiv.org/abs/1505.01504
    """
    output = list()
    output.append(0)
    for idx,v in enumerate(sequence):
        # note that idx starts at 0 for sequence and below we're appending to output so this is correct
        # behavior once we remove output[0] i.e. the first element will be the first element of the sequence
        output.append(alpha * output[idx] + v)
    output = output[1:]
    return output
    
    
    
"""
Build data set:
- Extract all sentences
- Build dictionary of words
- Include a special token for END_SEQ and BEGIN_SEQ?
- Build one-hot encoding dictionary using length of dictionary as length of vector

- For each sentence:
    - use biFOFE to build list of FOFE representations for each word in the sentence
    - targets for each training example are the one-hot encoding of the next word in the sequence (or the one hot of END_SEQ, BEGIN_SEQ)

split this up into training, validation, and test
"""
