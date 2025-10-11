import pickle, nltk, numpy, pandas as pd
from nltk.corpus import stopwords, twitter_samples
from os import getcwd
from utils.CosineSimilarity import cosine_similarity
from Classification.NaiveBayes import process_tweet
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

filePath = f"{getcwd()}/data/"
nltk.data.path.append(filePath)

class NaiveMachineTranslation():
    _X_train = None
    _Y_train = None
    _X_val = None
    _Y_val = None
    _R_train = None
    _en_embeddings_subset = None
    _fr_embeddings_subset = None
    _ind2Tweet = None
    _document_vecs = None
    _learning_rate:float = None
    _epochs: int = None

    def __init__(self, learning_rate:float, epochs:int):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._PrepareData();
    
    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        self._en_embeddings_subset = pickle.load(open("./data/en_embeddings.p", "rb"))
        self._fr_embeddings_subset = pickle.load(open("./data/fr_embeddings.p", "rb"))
        # loading the english to french dictionaries
        en_fr_train = self._get_dict('./data/en-fr.train.txt')
        print('The length of the English to French training dictionary is', len(en_fr_train))
        en_fr_test = self._get_dict('./data/en-fr.test.txt')
        print('The length of the English to French test dictionary is', len(en_fr_test))
        # getting the training set:
        self._X_train, self._Y_train = self._get_matrices(
            en_fr_train, self._fr_embeddings_subset, self._en_embeddings_subset)
        self._X_val, self._Y_val = self._get_matrices(en_fr_test, self._fr_embeddings_subset, self._en_embeddings_subset)
        print(f"X_train: {self._X_train.shape} Y_train: {self._Y_train.shape} X_val: {self._X_val.shape} Y_val: {self._Y_val.shape}")

    def TrainModel(self, verbose=True):
        '''
        Inputs:
            X: a matrix of dimension (m,n) where the columns are the English embeddings.
            Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
            train_steps: positive int - describes how many steps will gradient descent algorithm do.
            learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
        Outputs:
            R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
        '''
        print(f"\n=== {self.TrainModel.__name__} ===")
        # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
        # R is a square matrix with length equal to the number of dimensions in th  word embedding
        self._R_train = rng.random((self._X_train.shape[1], self._X_train.shape[1]))
        for i in range(self._epochs):
            if verbose and i % 25 == 0:
                print(f"loss at iteration {i} is: {self._compute_loss(self._X_train, self._Y_train, self._R_train):.4f}")
            # use the function that you defined to compute the gradient
            gradient = self._compute_gradient(self._X_train, self._Y_train, self._R_train)

            # update R by subtracting the learning rate times gradient
            self._R_train -= self._learning_rate * gradient

    def EvaluateModel(self):
        '''
        Input:
            X: a matrix where the columns are the English embeddings.
            Y: a matrix where the columns correspong to the French embeddings.
            R: the transform matrix which translates word embeddings from
            English to French word vector space.
        Output:
            accuracy: for the English to French capitals
        '''
        print(f"\n=== {self.EvaluateModel.__name__} ===")
        # The prediction is X times R
        pred = self._X_val @ self._R_train

        # initialize the number correct to zero
        num_correct = 0

        # loop through each row in pred (each transformed embedding)
        for i in range(len(pred)):
            # get the index of the nearest neighbor of pred at row 'i'; also pass in the candidates in Y
            pred_idx = self._nearest_neighbor(pred[i], self._Y_val)

            # if the index of the nearest neighbor equals the row of i... \
            if pred_idx == i:
                # increment the number correct by 1.
                num_correct += 1

        # accuracy is the number correct divided by the number of rows in 'pred' (also number of rows in X)
        accuracy = num_correct / len(pred)
        print(f"Model accuracy: {accuracy}")
    
    def _nearest_neighbor(self, v, candidates, k=1):
        """
        Input:
        - v, the vector you are going find the nearest neighbor for
        - candidates: a set of vectors where we will find the neighbors
        - k: top k nearest neighbors to find
        Output:
        - k_idx: the indices of the top k closest vectors in sorted form
        """
        similarity_l = []

        # for each candidate vector...
        for row in candidates:
            # get the cosine similarity
            cos_similarity = cosine_similarity(v, row)

            # append the similarity to the list
            similarity_l.append(cos_similarity)

        # sort the similarity list and get the indices of the sorted list    
        sorted_ids = numpy.argsort(similarity_l)
        
        # Reverse the order of the sorted_ids array
        sorted_ids = sorted_ids[::-1]
        
        # get the indices of the k most similar candidate vectors
        k_idx = sorted_ids[:k]
        return k_idx
    
    def _get_dict(self, path: str):
        data = pd.read_csv(path, delimiter=' ')
        print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
        print("\ndata.describe():")
        print(data.describe())
        print("\ndata.info():")
        data.info()
        etof = {}  # the english to french dictionary to be returned
        for i in range(len(data)):
            # indexing into the rows.
            en = data.iloc[i,0]
            fr = data.iloc[i,1]
            #print(f"{en} -> {fr}")
            etof[en] = fr
        return etof

    def _get_matrices(self, en_fr, french_vecs, english_vecs):
        """
        Translating English dictionary to French by using embeddings.
        Creates matrices of word embeddings for English and French words that are mapped to each other.
        
        Inputs:
            en_fr: Dictionary mapping English words to French words.
            french_vecs: Dictionary of French word embeddings.
            english_vecs: Dictionary of English word embeddings.
        
        Outputs: 
            X: Matrix with each row being the embedding of an English word. Shape is (number_of_words, embedding_size).
            Y: Matrix with each row being the embedding of the corresponding French word. Shape matches X.
        
        Note:
            This function does not compute or return a projection matrix.
        """
        # X_l and Y_l are lists of the english and french word embeddings
        X_l = list()
        Y_l = list()

        # get the english words (the keys in the dictionary) and store in a set()
        english_set = english_vecs.keys()

        # get the french words (keys in the dictionary) and store in a set()
        french_set = french_vecs.keys()

        # store the french words that are part of the english-french dictionary (these are the values of the dictionary)
        french_words = set(en_fr.values())

        # loop through all english, french word pairs in the english french dictionary
        for en_word, fr_word in en_fr.items():

            # check that the french word has an embedding and that the english word has an embedding
            if fr_word in french_set and en_word in english_set:

                # get the english embedding
                en_vec = english_vecs[en_word]

                # get the french embedding
                fr_vec = french_vecs[fr_word]

                # add the english embedding to the list
                X_l.append(en_vec)

                # add the french embedding to the list
                Y_l.append(fr_vec)
        # stack the vectors of X_l into a matrix X
        X = numpy.array(X_l)

        # stack the vectors of Y_l into a matrix Y
        Y = numpy.array(Y_l)
        return X, Y

    def _compute_loss(self, X, Y, R):
        '''
        squared Frobenius norm of the difference between matrix and its approximation, divided by the number of training examples m
    
        Inputs: 
            X: a matrix of dimension (m,n) where the columns are the English embeddings.
            Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
            R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
        Outputs:
            L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
        '''
        # m is the number of rows in X
        m = X.shape[0]
            
        # diff is XR - Y    
        diff = X @ R - Y

        # diff_squared is the element-wise square of the difference    
        diff_squared = diff ** 2

        # sum_diff_squared is the sum of the squared elements
        sum_diff_squared = numpy.sum(diff_squared)

        # loss i is the sum_diff_squared divided by the number of examples (m)
        return sum_diff_squared / m

    def _compute_gradient(self, X, Y, R):
        '''
        Inputs: 
            X: a matrix of dimension (m,n) where the columns are the English embeddings.
            Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
            R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
        Outputs:
            g: a scalar value - gradient of the loss function L for given X, Y and R.
        '''
        # m is the number of rows in X
        m = X.shape[0]

        # gradient is X^T(XR - Y) * 2/m    
        gradient = X.T @ (X @ R - Y) * 2/ m
        return gradient
   
if __name__ == "__main__":
    mt = NaiveMachineTranslation(0.01, 1000)
    mt.TrainModel(True)
    mt.EvaluateModel()