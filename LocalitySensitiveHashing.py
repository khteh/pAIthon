import pdb, pickle, string, time, nltk, numpy, pandas as pd
from nltk.corpus import stopwords, twitter_samples
from os import getcwd
from utils.CosineSimilarity import cosine_similarity
from Classification.NaiveBayes import process_tweet
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

filePath = f"{getcwd()}/../data/"
nltk.data.path.append(filePath)

class LocalitySensitiveHashing():
    """
    Implement a more efficient version of k-nearest neighbors using locality sensitive hashing
    """
    _en_embeddings_subset = None
    _all_positive_tweets = None
    _all_negative_tweets = None
    _all_tweets = None
    _n_planes: int = None
    _planes_l = None
    _universes: int = None
    _hash_tables = None
    _id_tables = None
    def __init__(self, planes: int, universes:int):
        self._n_planes = planes
        self._universes = universes
        self._PrepareData()

    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        # get the positive and negative tweets
        self._en_embeddings_subset = pickle.load(open("./data/en_embeddings.p", "rb"))
        self._all_positive_tweets = twitter_samples.strings('positive_tweets.json')
        self._all_negative_tweets = twitter_samples.strings('negative_tweets.json')
        self._all_tweets = self._all_positive_tweets + self._all_negative_tweets
        #tweet_embedding = self._get_document_embedding(custom_tweet, en_embeddings_subset)
        self._document_vecs, self._ind2Tweet = self._get_document_vecs(self._all_tweets, self._en_embeddings_subset)
        self._planes_l = [numpy.random.normal(size=(len(self._ind2Tweet[1]), self._n_planes)) for _ in range(self._universes)]
        self._create_hash_id_tables(self._universes)
        print(f"length of dictionary {len(self._ind2Tweet)}")
        print(f"shape of document_vecs {self._document_vecs.shape}")

    def GetSimilarExistingTweet(self, tweet:str):
        process_tweet(tweet)
        tweet_embedding = self._get_document_embedding(tweet, self._en_embeddings_subset)
        idx = numpy.argmax(cosine_similarity(self._document_vecs, tweet_embedding))
        return self._all_tweets[idx]

    def SearchDocuments(self, id: int, k: int, universe_count: int):
        doc_to_search = self._all_tweets[id]
        vec_to_search = self._document_vecs[id]        
        nearest_neighbor_ids = self._approximate_knn(id, vec_to_search, k, universe_count)
        print(f"Nearest neighbors for document {id}")
        print(f"Document contents: {doc_to_search}")
        print("")
        for neighbor_id in nearest_neighbor_ids:
            print(f"Nearest neighbor at document id {neighbor_id}")
            print(f"document contents: {self._all_tweets[neighbor_id]}")

    def _approximate_knn(self, doc_id, v, k, num_universes_to_use):
        """Search for k-NN using hashes."""
        #assert num_universes_to_use <= N_UNIVERSES

        # Vectors that will be checked as possible nearest neighbor
        vecs_to_consider_l = list()

        # list of document IDs
        ids_to_consider_l = list()

        # create a set for ids to consider, for faster checking if a document ID already exists in the set
        ids_to_consider_set = set()

        # loop through the universes of planes
        for universe_id in range(num_universes_to_use):

            # get the set of planes from the planes_l list, for this particular universe_id
            planes = self._planes_l[universe_id]

            # get the hash value of the vector for this set of planes
            hash_value = self._hash_value_of_vector(v, planes)

            # get the hash table for this particular universe_id
            hash_table = self._hash_tables[universe_id]

            # get the list of document vectors for this hash table, where the key is the hash_value
            document_vectors_l = hash_table[hash_value]

            # get the id_table for this particular universe_id
            id_table = self._id_tables[universe_id]

            # get the subset of documents to consider as nearest neighbors from this id_table dictionary
            new_ids_to_consider = id_table[hash_value]

            ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

            # loop through the subset of document vectors to consider
            for i, new_id in enumerate(new_ids_to_consider):
                
                if doc_id == new_id:
                    continue

                # if the document ID is not yet in the set ids_to_consider...
                if new_id not in ids_to_consider_set:
                    # access document_vectors_l list at index i to get the embedding
                    # then append it to the list of vectors to consider as possible nearest neighbors
                    document_vector_at_i = document_vectors_l[i]
                    vecs_to_consider_l.append(document_vector_at_i)

                    # append the new_id (the index for the document) to the list of ids to consider
                    ids_to_consider_l.append(new_id)

                    # also add the new_id to the set of ids to consider
                    # (use this to check if new_id is not already in the IDs to consider)
                    ids_to_consider_set.add(new_id)

        # Now run k-NN on the smaller set of vecs-to-consider.
        print("Fast considering %d vecs" % len(vecs_to_consider_l))

        # convert the vecs to consider set to a list, then to a numpy array
        vecs_to_consider_arr = numpy.array(vecs_to_consider_l)

        # call nearest neighbors on the reduced list of candidate vectors
        nearest_neighbor_idx_l = self._nearest_neighbor(v, vecs_to_consider_arr, k=k)

        # Use the nearest neighbor index list as indices into the ids to consider
        # create a list of nearest neighbors by the document ids
        nearest_neighbor_ids = [ids_to_consider_l[idx]
                                for idx in nearest_neighbor_idx_l]
        return nearest_neighbor_ids
    
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

    def _get_document_embedding(self, tweet, en_embeddings, process_tweet=process_tweet):
        '''
        Input:
            - tweet: a string
            - en_embeddings: a dictionary of word embeddings
        Output:
            - doc_embedding: sum of all word embeddings in the tweet
        '''
        doc_embedding = numpy.zeros(300)
        # process the document into a list of words (process the tweet)
        processed_doc = process_tweet(tweet)
        for word in processed_doc:
            # add the word embedding to the running total for the document embedding
            doc_embedding += en_embeddings.get(word, 0)
        return doc_embedding   

    def _get_document_vecs(self, all_docs, en_embeddings):
        '''
        Input:
            - all_docs: list of strings - all tweets in our dataset.
            - en_embeddings: dictionary with words as the keys and their embeddings as the values.
        Output:
            - document_vec_matrix: matrix of tweet embeddings.
            - ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values.
        '''
        # the dictionary's key is an index (integer) that identifies a specific tweet
        # the value is the document embedding for that document
        ind2Doc_dict = {}

        # this is list that will store the document vectors
        document_vec_l = []

        for i, doc in enumerate(all_docs):

            ### START CODE HERE ###
            # get the document embedding of the tweet
            doc_embedding = self._get_document_embedding(doc, en_embeddings)

            # save the document embedding into the ind2Tweet dictionary at index i
            ind2Doc_dict[i] = doc_embedding

            # append the document embedding to the list of document vectors
            document_vec_l.append(doc_embedding)

        # convert the list of document vectors into a 2D array (each row is a document vector)
        document_vec_matrix = numpy.vstack(document_vec_l)
        return document_vec_matrix, ind2Doc_dict

    def _hash_value_of_vector(self, v, planes):
        """Create a hash for a vector; hash_id says which random hash to use.
        Input:
            - v:  vector of tweet. It's dimension is (1, N_DIMS)
            - planes: matrix of dimension (N_DIMS, N_n_planes) - the set of planes that divide up the region
        Output:
            - res: a number which is used as a hash for your vector

        """
        ### START CODE HERE ###
        # for the set of planes,
        # calculate the dot product between the vector and the matrix containing the planes
        # remember that planes has shape (300, 10)
        # The dot product will have the shape (1,10)    
        dot_product = v @ planes
            
        # get the sign of the dot product (1,10) shaped vector
        sign_of_dot_product = numpy.sign(dot_product)

        # set h to be false (eqivalent to 0 when used in operations) if the sign is negative,
        # and true (equivalent to 1) if the sign is positive (1,10) shaped vector
        # if the sign is 0, i.e. the vector is in the plane, consider the sign to be positive
        h = sign_of_dot_product >= 0

        # remove extra un-used dimensions (convert this from a 2D to a 1D array)
        h = numpy.squeeze(h)

        # initialize the hash value to 0
        hash_value = 0

        n_n_planes = len(h)
        for i in range(n_n_planes):
            # increment the hash value by 2^i * h_i        
            hash_value += 2 ** i * h[i]
            
        # cast hash_value as an integer
        return int(hash_value)

    def _make_hash_table(self, vecs, planes):
        """
        Input:
            - vecs: list of vectors to be hashed.
            - planes: the matrix of planes in a single "universe", with shape (embedding dimensions, number of planes).
        Output:
            - hash_table: dictionary - keys are hashes, values are lists of vectors (hash buckets)
            - id_table: dictionary - keys are hashes, values are list of vectors id's
                                (it's used to know which tweet corresponds to the hashed vector)
        """
        # number of planes is the number of columns in the planes matrix
        num_of_planes = planes.shape[1]

        # number of buckets is 2^(number of planes)
        # ALTERNATIVE SOLUTION COMMENT:
        # num_buckets = pow(2, num_of_planes)
        num_buckets = 2**num_of_planes

        # create the hash table as a dictionary.
        # Keys are integers (0,1,2.. number of buckets)
        # Values are empty lists
        hash_table = {i: [] for i in range(num_buckets)}

        # create the id table as a dictionary.
        # Keys are integers (0,1,2... number of buckets)
        # Values are empty lists
        id_table = {i: [] for i in range(num_buckets)}

        # for each vector in 'vecs'
        for i, v in enumerate(vecs):
            # calculate the hash value for the vector
            h = self._hash_value_of_vector(v, planes)

            # store the vector into hash_table at key h,
            # by appending the vector v to the list at key h
            hash_table[h].append(v) # @REPLACE None

            # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
            # the key is the h, and the 'i' is appended to the list at key h
            id_table[h].append(i) # @REPLACE None
        return hash_table, id_table
    
    def _create_hash_id_tables(self, n_universes):
        self._hash_tables = []
        self._id_tables = []
        for universe_id in range(n_universes):  # there are 25 hashes
            print('working on hash universe #:', universe_id)
            planes = self._planes_l[universe_id]
            hash_table, id_table = self._make_hash_table(self._document_vecs, planes)
            self._hash_tables.append(hash_table)
            self._id_tables.append(id_table)

if __name__ == "__main__":
    lsh = LocalitySensitiveHashing(10, 25)
    tweet = "I am sad!"
    print(f"Simiar tweet for '{tweet}' is '{lsh.GetSimilarExistingTweet(tweet)}'")
    lsh.SearchDocuments(0, 3, 5)