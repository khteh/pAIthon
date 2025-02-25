import numpy as np
"""
Model that represents text as an unordered collection of words.
This model can be used to build Naive Bayes classifier which classifies text based on Bayes' Rule
Naive because of the assumption that the words are independent of each other
Example: P(Happy | "I", "love", "it") = P("I", "love", "it" | Happy)P(Happy) / P("I", "love", "it")
                                      ~= P("I", "love", "it" | Happy)P(Happy) <= Proportional to
                                      ~= P("I", "love", "it", Happy) <= Proportional to this joint probability
                                      ~= P(Happy)P("I"|Happy)P("love"|Happy)P("it"|Happy) <= Naively proportional to
word2vec - Model for generating word vectors
        - Example: Calculate distance/closeness between words, get a list of similar / nearest words, show relationships between words.
"You shall know a word by the company it keeps" (J. R. Firth, 1957)

Transformer:
Encoder:
Input word -> Positional Encoding -> [Self Attention (Multiple) -> NN (Feedforward)] -> Encoded representation
[Self Attention (Multiple) -> NN] could be applied multiple times

Decoder:
Previous output word -> Positional Encoding -> [Self Attention (Multiple) -> Attention (Multiple) -> NN (Feedforward)] -> Next output word
                                                    Encoded representations ->
"""
data = [[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 3, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]]

def distance(row1, row2):
    # fix this function so that it returns 
    # the sum of differences between the occurrences
    # of each word in row1 and row2.
    # you can assume that row1 and row2 are lists with equal length, containing numeric values.
    distance = [abs(i- j) for i, j in zip(row1, row2)]
    if sum(distance) == 0:
        return [np.inf]*len(row1)
    return distance
    
def all_pairs(data):
    # this calls the distance function for all the two-row combinations in the data
    # you do not need to change this
    dist = [[distance(sent1, sent2) for sent1 in data] for sent2 in data]
    print(dist)

def find_nearest_pair(data):
    N = len(data)
    dist = np.empty((N, N), dtype=float)
    #print(f"dist: {dist}")
    #dist = np.array([[distance(sent1, sent2) for sent1 in data] for sent2 in data], dtype=float)
    #print(f"dist: {dist}")
    for i in range(0,N):
        for j in range(0, N):
            if i != j:
                dist[i][j] = sum([abs(i- j) for i, j in zip(data[i], data[j])])
            else:
                dist[i][j] = np.inf
    '''
A quick way to get the index of the element with the lowest value in a 2D array (or in fact, any dimension) is by the function
np.unravel_index(np.argmin(dist), dist.shape))
where dist is the 2D array. This will return the index as a list of length two. If you're curious, here's an intuitive explanation of the function: https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index
and here's its documentation. https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
    '''
    print(np.unravel_index(np.argmin(dist), dist.shape))

if __name__ == "__main__":
    find_nearest_pair(data)
