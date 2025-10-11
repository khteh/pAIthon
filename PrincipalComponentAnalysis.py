import numpy, pickle
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

def get_vectors(embeddings, words):
    """
    Input:
        embeddings: a word 
        fr_embeddings:
        words: a list of words
    Output: 
        X: a matrix where the rows are the embeddings corresponding to the rows on the list
        
    """
    m = len(words)
    X = numpy.zeros((1, 300))
    for word in words:
        english = word
        eng_emb = embeddings[english]
        X = numpy.vstack((X, eng_emb))
    X = X[1:,:]
    return X

def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    # mean center the data
    X_demeaned = X - numpy.mean(X)

    # calculate the covariance matrix
    covariance_matrix = numpy.cov(X.T) # Each row of m represents a variable, and each column a single observation of all those variables.

    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = numpy.linalg.eigh(covariance_matrix)

    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = numpy.argsort(eigen_vals)
    
    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[idx_sorted_decreasing]

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or n_components)
    eigen_vecs_subset = eigen_vecs_sorted[:n_components]

    # transform the data by multiplying the transpose of the eigenvectors with the transpose of the de-meaned data
    # Then take the transpose of that product. Note that, since for any matrices A, B, (A.B).T = B.T . A.T,
    # this reduces to the dot product of the de-mean data with the eigenvectors
    X_reduced = X_demeaned @ eigen_vecs_sorted
    return X_reduced

if __name__ == "__main__":
    X = rng.random((3, 10))
    X_reduced = compute_pca(X, n_components=2)
    print(f"Original matrix was {X.shape} and it became: {X_reduced}")
    words = ['oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']

    word_embeddings = pickle.load(open("./data/word_embeddings_subset.p", "rb")) # Complete dataset available at https://code.google.com/archive/p/word2vec/
    print(f"{len(word_embeddings)} embeddings")  # there should be 243 words that will be used in this assignment
    # given a list of words and the embeddings, it returns a matrix with all the embeddings
    X = get_vectors(word_embeddings, words)

    print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)
    result = compute_pca(X, 2)
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

    plt.show()