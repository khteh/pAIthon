import math, re, random, numpy, nltk, pandas as pd
from pandas import DataFrame
nltk.data.path.append('.')

class AutoCompletion():
    _path: str = None
    _data = None
    _sentences = None
    _tokens = None
    _word_counts = None
    _start_token:str = None
    _end_token:str = None
    _unknown_token: str = None
    _threshold:int = None
    _vocabulary = None
    _train_data = None
    _test_data = None
    def __init__(self, path:str, unknown_token:str, start_token:str, end_token:str, threshold:int):
        self._path = path
        self._unknown_token = unknown_token
        self._start_token = start_token
        self._end_token = end_token
        self._threshold = threshold
        self._PrepareData()

    def suggest_a_word(self, previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
        """
        Get suggestion for the next word
        
        Args:
            previous_tokens: The sentence you input where each token is a word. Must have length >= n 
            n_gram_counts: Dictionary of counts of n-grams
            n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            vocabulary: List of words
            k: positive constant, smoothing parameter
            start_with: If not None, specifies the first few letters of the next word
            
        Returns:
            A tuple of 
            - string of the most likely next word
            - corresponding probability
        """
        
        # length of previous words
        n = len(list(n_gram_counts.keys())[0])
        
        # append "start token" on "previous_tokens"
        previous_tokens = [self._start_token] * n + previous_tokens
        
        # From the words that the user already typed
        # get the most recent 'n' words as the previous n-gram
        previous_n_gram = previous_tokens[-n:]

        # Estimate the probabilities that each word in the vocabulary
        # is the next word,
        # given the previous n-gram, the dictionary of n-gram counts,
        # the dictionary of n plus 1 gram counts, and the smoothing constant
        probabilities = self.estimate_probabilities(previous_n_gram,
                                            n_gram_counts, n_plus1_gram_counts,
                                            vocabulary, k=k)
        
        # Initialize suggested word to None
        # This will be set to the word with highest probability
        suggestion = None
        
        # Initialize the highest word probability to 0
        # this will be set to the highest probability 
        # of all words to be suggested
        max_prob = 0
        
        # For each word and its probability in the probabilities dictionary:
        for word, prob in probabilities.items(): # complete this line
            
            # If the optional start_with string is set
            if start_with: # complete this line with the proper condition
                
                # Check if the beginning of word does not match with the letters in 'start_with'
                if not word.startswith(start_with): # complete this line with the proper condition

                    # if they don't match, skip this word (move onto the next word)
                    continue
            
            # Check if this word's probability
            # is greater than the current maximum probability
            if prob > max_prob: # complete this line with the proper condition
                
                # If so, save this word as the best suggestion (so far)
                suggestion = word
                
                # Save the new maximum probability
                max_prob = prob
        return suggestion, max_prob

    def get_suggestions(self, previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
        model_counts = len(n_gram_counts_list)
        suggestions = []
        if not vocabulary:
            vocabulary = self._vocabulary
        for i in range(model_counts-1):
            n_gram_counts = n_gram_counts_list[i]
            n_plus1_gram_counts = n_gram_counts_list[i+1]
            suggestion = self.suggest_a_word(previous_tokens, n_gram_counts,
                                        n_plus1_gram_counts, vocabulary,
                                        k=k, start_with=start_with)
            suggestions.append(suggestion)
        return suggestions    
    
    def estimate_probability(self, word, previous_n_gram, 
                            n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
        """
        Estimate the probabilities of a next word using the n-gram counts with k-smoothing
        
        Args:
            word: next word
            previous_n_gram: A sequence of words of length n
            n_gram_counts: Dictionary of counts of n-grams
            n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            vocabulary_size: number of words in the vocabulary
            k: positive constant, smoothing parameter
        
        Returns:
            A probability
        """
        #print(f"\n=== {self.estimate_probability.__name__} ===")
        # convert list to tuple to use it as a dictionary key
        previous_n_gram = tuple(previous_n_gram)
        
        # Set the denominator
        # If the previous n-gram exists in the dictionary of n-gram counts,
        # Get its count.  Otherwise set the count to zero
        # Use the dictionary that has counts for n-grams
        previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
                
        # Calculate the denominator using the count of the previous n gram
        # and apply k-smoothing
        denominator = previous_n_gram_count + k * vocabulary_size

        # Define n plus 1 gram as the previous n-gram plus the current word as a tuple
        n_plus1_gram = previous_n_gram + (word,)
    
        # Set the count to the count in the dictionary,
        # otherwise 0 if not in the dictionary
        # use the dictionary that has counts for the n-gram plus current word    
        n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
                
        # Define the numerator use the count of the n-gram plus current word,
        # and apply smoothing
        numerator = n_plus1_gram_count + k
            
        # Calculate the probability as the numerator divided by denominator
        return numerator / denominator
    
    def estimate_probabilities(self, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary,  k=1.0):
        """
        Estimate the probabilities of next words using the n-gram counts with k-smoothing
        
        Args:
            previous_n_gram: A sequence of words of length n
            n_gram_counts: Dictionary of counts of n-grams
            n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            vocabulary: List of words
            k: positive constant, smoothing parameter
        
        Returns:
            A dictionary mapping from next words to the probability.
        """
        #print(f"\n=== {self.estimate_probabilities.__name__} ===")
        # convert list to tuple to use it as a dictionary key
        previous_n_gram = tuple(previous_n_gram)    
        
        # add <e> <unk> to the vocabulary
        # <s> is not needed since it should not appear as the next word
        vocabulary = vocabulary + [self._end_token, self._unknown_token]    
        vocabulary_size = len(vocabulary)    
        
        probabilities = {}
        for word in vocabulary:
            probability = self.estimate_probability(word, previous_n_gram, 
                                            n_gram_counts, n_plus1_gram_counts, 
                                            vocabulary_size, k=k)
            probabilities[word] = probability
        return probabilities
    
    def make_count_matrix(self, n_plus1_gram_counts, vocabulary) -> DataFrame:
        print(f"\n=== {self.make_count_matrix.__name__} ===")
        # add <e> <unk> to the vocabulary
        # <s> is omitted since it should not appear as the next word
        vocabulary = vocabulary + [self._end_token, self._unknown_token]
        
        # obtain unique n-grams
        n_grams = []
        for n_plus1_gram in n_plus1_gram_counts.keys():
            n_gram = n_plus1_gram[0:-1]        
            n_grams.append(n_gram)
        n_grams = list(set(n_grams))
        
        # mapping from n-gram to row
        row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}    
        # mapping from next word to column
        col_index = {word:j for j, word in enumerate(vocabulary)}    
        
        nrow = len(n_grams)
        ncol = len(vocabulary)
        count_matrix = numpy.zeros((nrow, ncol))
        for n_plus1_gram, count in n_plus1_gram_counts.items():
            n_gram = n_plus1_gram[0:-1]
            word = n_plus1_gram[-1]
            if word not in vocabulary:
                continue
            i = row_index[n_gram]
            j = col_index[word]
            count_matrix[i, j] = count
        
        count_matrix = DataFrame(count_matrix, index=n_grams, columns=vocabulary)
        return count_matrix
    
    def make_probability_matrix(self, n_plus1_gram_counts, vocabulary, k) -> DataFrame:
        print(f"\n=== {self.make_probability_matrix.__name__} ===")
        count_matrix = self.make_count_matrix(n_plus1_gram_counts, vocabulary)
        count_matrix += k
        return count_matrix.div(count_matrix.sum(axis=1), axis=0)
    
    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        with open(self._path, "r") as f:
            self._data = f.read()
        print("Data type:", type(self._data))
        print("Number of letters:", len(self._data))
        self._split_to_sentences()
        self._tokenize_sentences()
        self._count_words()
        # Get the closed vocabulary using the train data
        self._vocabulary = self._get_words_with_nplus_frequency()
        
        # For the train data, replace less common words with "<unk>"
        self._train_data = self._replace_oov_words_by_unk()
        
        # For the test data, replace less common words with "<unk>"
        self._test_data = self._replace_oov_words_by_unk()
        print(f"First preprocessed training sample: {self._train_data[0]}")
        print(f"First preprocessed test sample: {self._test_data[0]}")
        print(f"Size of vocabulary: {len(self._vocabulary)}")
        print(f"First 10 vocabulary: {self._vocabulary[0:10]}")

    def count_n_grams(self, tokens, n):
        """
        Count all n-grams in the data
        
        Args:
            data: List of lists of words
            n: number of words in a sequence
        
        Returns:
            A dictionary that maps a tuple of n-words to its frequency
            Use tuple instead of a list as the dictionary key because tuple is immutable. List is mutable.
            Use n starting markers instead of n-1 in order to use them to compute initial probability for tne (n+1)-gram.
        """
        #print(f"\n=== {self.count_n_grams.__name__} ===")
        # Initialize dictionary of n-grams and their counts
        n_grams = {}
        if not tokens:
            tokens = self._tokens
        # Go through each sentence in the data
        for sentence in tokens: # complete this line
            
            # prepend start token n times, and  append the end token one time
            s = sentence.copy()
            s[:0] = [self._start_token] * n
            s.append(self._end_token)
            #print(f"sentence: {s}")
            
            # convert list to tuple
            # So that the sequence of words can be used as
            # a key in the dictionary
            s = tuple(s)
            
            # Use 'i' to indicate the start of the n-gram
            # from index 0
            # to the last index where the end of the n-gram
            # is within the sentence.
            for i in range(len(s)-n+1): # complete this line

                # Get the n-gram from i to i+n
                n_gram = s[i:i+n]
                
                # check if the n-gram is in the dictionary
                if n_gram in n_grams: # complete this line with the proper condition
                
                    # Increment the count for this n-gram
                    n_grams[n_gram] += 1
                else:
                    # Initialize this n-gram count to 1
                    n_grams[n_gram] = 1
        return n_grams

    def calculate_perplexity(self, sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0) -> float:
        """
        Calculate perplexity for a list of sentences
        
        Args:
            sentence: List of strings
            n_gram_counts: Dictionary of counts of n-grams
            n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            vocabulary_size: number of unique words in the vocabulary
            k: Positive smoothing constant
        
        Returns:
            Perplexity score
        """
        print(f"\n=== {self.calculate_perplexity.__name__} ===")
        # length of previous words
        n = len(list(n_gram_counts.keys())[0]) 
        
        # prepend <s> and append <e>
        sentence = [self._start_token] * n + sentence + [self._end_token]
        
        # Cast the sentence from a list to a tuple
        sentence = tuple(sentence)
        
        # length of sentence (after adding <s> and <e> tokens)
        N = len(sentence)
        
        # The variable p will hold the product
        # that is calculated inside the n-root
        # Update this in the code below
        product_pi = 1.0
        print(f"n: {n}, N: {N}, sentence: {sentence}")
        # Index t ranges from n to N - 1, inclusive on both ends
        for t in range(n, N):

            # get the n-gram preceding the word at position t
            n_gram = sentence[t-n:t]

            # get the word at position t
            word = sentence[t]
            
            # Estimate the probability of the word given the n-gram
            # using the n-gram counts, n-plus1-gram counts,
            # vocabulary size, and smoothing constant
            # def estimate_probability(word, previous_n_gram, 
            #                 n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
            probability = self.estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)
            #print(f"n_gram: {n_gram}, word: {word} probability: {probability}")
            # Update the product of the probabilities
            # This 'product_pi' is a cumulative product 
            # of the (1/P) factors that are calculated in the loop
            product_pi *= (1/probability)

        # Take the Nth root of the product
        perplexity = (product_pi)**(1/N)
        return perplexity
    
    def _split_to_sentences(self):
        """
        Split data by linebreak "\n"
        
        Args:
            data: str
        
        Returns:
            A list of sentences
        """
        self._sentences = self._data.split('\n')
        
        # Additional clearning (This part is already implemented)
        # - Remove leading and trailing spaces from each sentence
        # - Drop sentences if they are empty strings.
        self._sentences = [s.strip() for s in self._sentences]
        self._sentences = [s for s in self._sentences if len(s) > 0]

    def _tokenize_sentences(self):
        """
        Tokenize sentences into tokens (words)
        
        Args:
            sentences: List of strings
        
        Returns:
            List of lists of tokens
        """
        # Initialize the list of lists of tokenized sentences
        self._tokens = []
        ### START CODE HERE ###
        
        # Go through each sentence
        for sentence in self._sentences: # complete this line
            
            # Convert into a list of words
            string_regex = re.compile('|'.join([
                r'(\w+)\s*([.,!?;]*)',
            ]))
            result = re.findall(string_regex, sentence.lower())
            tokens = [x for t in result for x in t if x]
            #tokens = re.split(r'(\w+)([.,!?;])*', sentence.strip().lower())
            
            # append the list of words to the list of lists
            self._tokens.append(tokens)

    def _count_words(self):
        """
        Count the number of word appearence in the tokenized sentences
        
        Args:
            tokenized_sentences: List of lists of strings
        
        Returns:
            dict that maps word (str) to the frequency (int)
        """
        self._word_counts = {}
        
        # Loop through each sentence
        for sentence in self._tokens: # complete this line
            
            # Go through each token in the sentence
            for token in sentence: # complete this line

                # If the token is not in the dictionary yet, set the count to 1
                if token not in self._word_counts: # complete this line with the proper condition
                    self._word_counts[token] = 1
                
                # If the token is already in the dictionary, increment the count by 1
                else:
                    self._word_counts[token] += 1

    def _get_words_with_nplus_frequency(self):
        """
        Find the words that appear N times or more
        
        Args:
            tokenized_sentences: List of lists of sentences
            count_threshold: minimum number of occurrences for a word to be in the closed vocabulary.
        
        Returns:
            List of words that appear N times or more
        """
        # Initialize an empty list to contain the words that
        # appear at least 'minimum_freq' times.
        
        # Get the word couts of the tokenized sentences
        # Use the function that you defined earlier to count the words
        #word_counts = self._count_words(tokenized_sentences) This is done in _PrepareData()
        return [k for k,v in self._word_counts.items() if v >= self._threshold]

    def _replace_oov_words_by_unk(self):
        """
        Replace words not in the given vocabulary with '<unk>' token.
        
        Args:
            tokenized_sentences: List of lists of strings
            vocabulary: List of strings that we will use
            unknown_token: A string representing unknown (out-of-vocabulary) words
        
        Returns:
            List of lists of strings, with words not in the vocabulary replaced
        """
        # Place vocabulary into a set for faster search
        vocabulary = set(self._vocabulary)
        
        # Initialize a list that will hold the sentences
        # after less frequent words are replaced by the unknown token
        replaced_tokenized_sentences = []
        
        # Go through each sentence
        for sentence in self._tokens:
            
            # Initialize the list that will contain
            # a single sentence with "unknown_token" replacements
            replaced_sentence = [t if t in vocabulary else self._unknown_token for t in sentence]
            
            # Append the list of tokens to the list of lists
            replaced_tokenized_sentences.append(replaced_sentence)
        return replaced_tokenized_sentences

def count_n_grams_tests():
    # test your code
    print(f"\n=== {count_n_grams_tests.__name__} ===")
    autocomplete = AutoCompletion("./data/en_US.twitter.txt", "<unk>", "<s>", "<e>", 1)
    sentences = [['i', 'like', 'a', 'cat'],
                ['this', 'dog', 'is', 'like', 'a', 'cat']]
    print("Uni-gram:")
    print(autocomplete.count_n_grams(sentences, 1))
    print("Bi-gram:")
    print(autocomplete.count_n_grams(sentences, 2))    

def estimate_probability_tests():
    print(f"\n=== {estimate_probability_tests.__name__} ===")
    # test your code
    autocomplete = AutoCompletion("./data/en_US.twitter.txt", "<unk>", "<s>", "<e>", 1)
    sentences = [['i', 'like', 'a', 'cat'],
                ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))

    unigram_counts = autocomplete.count_n_grams(sentences, 1)
    bigram_counts = autocomplete.count_n_grams(sentences, 2)
    probability = autocomplete.estimate_probability("cat", ["a"], unigram_counts, bigram_counts, len(unique_words), k=1)
    print(f"The estimated probability of word 'cat' given the previous n-gram 'a' is: {probability:.4f}")

def estimate_probabilities_tests():
    print(f"\n=== {estimate_probabilities_tests.__name__} ===")
    # test your code
    autocomplete = AutoCompletion("./data/en_US.twitter.txt", "<unk>", "<s>", "<e>", 1)
    sentences = [['i', 'like', 'a', 'cat'],
                ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))
    unigram_counts = autocomplete.count_n_grams(sentences, 1)
    bigram_counts = autocomplete.count_n_grams(sentences, 2)
    print(f"unigram_counts: {unigram_counts}, bigram_counts: {bigram_counts}")
    probabilities = autocomplete.estimate_probabilities(["a"], unigram_counts, bigram_counts, unique_words, k=1)
    print(f"probabilities: {probabilities}")
    # Additional test
    trigram_counts = autocomplete.count_n_grams(sentences, 3)
    print(f"trigram_counts: {trigram_counts}")
    probabilities = autocomplete.estimate_probabilities(["<s>", "<s>"], bigram_counts, trigram_counts, unique_words, k=1)
    print(f"probabilities: {probabilities}")

def count_matrix_tests():
    print(f"\n=== {count_matrix_tests.__name__} ===")
    autocomplete = AutoCompletion("./data/en_US.twitter.txt", "<unk>", "<s>", "<e>", 1)
    sentences = [['i', 'like', 'a', 'cat'],
                    ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))
    bigram_counts = autocomplete.count_n_grams(sentences, 2)
    data = autocomplete.make_count_matrix(bigram_counts, unique_words)
    print('bigram counts:')
    print(data)
    # Show trigram counts
    trigram_counts = autocomplete.count_n_grams(sentences, 3)
    data = autocomplete.make_count_matrix(trigram_counts, unique_words)
    print('\ntrigram counts:')
    print(data)

def probability_matrix_tests():
    print(f"\n=== {probability_matrix_tests.__name__} ===")
    autocomplete = AutoCompletion("./data/en_US.twitter.txt", "<unk>", "<s>", "<e>", 1)
    sentences = [['i', 'like', 'a', 'cat'],
                    ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))
    bigram_counts = autocomplete.count_n_grams(sentences, 2)
    data = autocomplete.make_probability_matrix(bigram_counts, unique_words, k=1)
    print("bigram probabilities:")
    print(data)
    trigram_counts = autocomplete.count_n_grams(sentences, 3)
    data = autocomplete.make_probability_matrix(trigram_counts, unique_words, k=1)
    print("\ntrigram probabilities:")
    print(data)

def perplexity_tests():
    print(f"\n=== {perplexity_tests.__name__} ===")
    autocomplete = AutoCompletion("./data/en_US.twitter.txt", "<unk>", "<s>", "<e>", 1)
    sentences = [['i', 'like', 'a', 'cat'],
                    ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))
    unigram_counts = autocomplete.count_n_grams(sentences, 1)
    bigram_counts = autocomplete.count_n_grams(sentences, 2)

    perplexity_train = autocomplete.calculate_perplexity(sentences[0],
                                            unigram_counts, bigram_counts,
                                            len(unique_words), k=1.0)
    print(f"Perplexity for first train sample: {perplexity_train:.4f}")

    test_sentence = ['i', 'like', 'a', 'dog']
    perplexity_test = autocomplete.calculate_perplexity(test_sentence,
                                        unigram_counts, bigram_counts,
                                        len(unique_words), k=1.0)
    print(f"Perplexity for test sample: {perplexity_test:.4f}")

def suggest_a_word_tests():
    print(f"\n=== {suggest_a_word_tests.__name__} ===")
    autocomplete = AutoCompletion("./data/en_US.twitter.txt", "<unk>", "<s>", "<e>", 1)
    # test your code
    sentences = [['i', 'like', 'a', 'cat'],
                ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))

    unigram_counts = autocomplete.count_n_grams(sentences, 1)
    bigram_counts = autocomplete.count_n_grams(sentences, 2)

    previous_tokens = ["i", "like"]
    tmp_suggest1 = autocomplete.suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0)
    print(f"The previous words are 'i like',\n\tand the suggested word is `{tmp_suggest1[0]}` with a probability of {tmp_suggest1[1]:.4f}")

    # test your code when setting the starts_with
    tmp_starts_with = 'c'
    tmp_suggest2 = autocomplete.suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0, start_with=tmp_starts_with)
    print(f"The previous words are 'i like', the suggestion must start with `{tmp_starts_with}`\n\tand the suggested word is `{tmp_suggest2[0]}` with a probability of {tmp_suggest2[1]:.4f}")    

def get_suggestions_tests():
    print(f"\n=== {get_suggestions_tests.__name__} ===")
    autocomplete = AutoCompletion("./data/en_US.twitter.txt", "<unk>", "<s>", "<e>", 1)
    # test your code
    sentences = [['i', 'like', 'a', 'cat'],
                ['this', 'dog', 'is', 'like', 'a', 'cat']]
    unique_words = list(set(sentences[0] + sentences[1]))

    unigram_counts = autocomplete.count_n_grams(sentences, 1)
    bigram_counts = autocomplete.count_n_grams(sentences, 2)
    trigram_counts = autocomplete.count_n_grams(sentences, 3)
    quadgram_counts = autocomplete.count_n_grams(sentences, 4)
    qintgram_counts = autocomplete.count_n_grams(sentences, 5)
    n_gram_counts_list = [unigram_counts, bigram_counts, trigram_counts, quadgram_counts, qintgram_counts]
    previous_tokens = ["i", "like"]
    suggestions = autocomplete.get_suggestions(previous_tokens, n_gram_counts_list, unique_words, k=1.0)
    print(f"The previous words are 'i like', the suggestions are:")
    print(suggestions)
    print(f"Get suggestions with varying n-grams...")
    n_gram_counts_list = []
    for n in range(1, 6):
        print("Computing n-gram counts with n =", n, "...")
        n_model_counts = autocomplete.count_n_grams(None, n)
        n_gram_counts_list.append(n_model_counts)
    previous_tokens = ["i", "am", "to"]
    suggestions = autocomplete.get_suggestions(previous_tokens, n_gram_counts_list, None, k=1.0)
    print(f"The previous words are {previous_tokens}, the suggestions are:")
    print(suggestions)
    previous_tokens = ["i", "want", "to", "go"]
    suggestions = autocomplete.get_suggestions(previous_tokens, n_gram_counts_list, None, k=1.0)
    print(f"\nThe previous words are {previous_tokens}, the suggestions are:")
    print(suggestions)
    previous_tokens = ["hey", "how", "are"]
    suggestions = autocomplete.get_suggestions(previous_tokens, n_gram_counts_list, None, k=1.0)
    print(f"\nThe previous words are {previous_tokens}, the suggestions are:")
    print(suggestions)
    previous_tokens = ["hey", "how", "are", "you"]
    suggestions = autocomplete.get_suggestions(previous_tokens, n_gram_counts_list, None, k=1.0)
    print(f"\nThe previous words are {previous_tokens}, the suggestions are:")
    print(suggestions)
if __name__ == "__main__":
    #autocomplete = AutoCompletion("./data/en_US.twitter.txt", "<unk>", "<s>", "<e>", 1)
    count_n_grams_tests()
    estimate_probability_tests()
    estimate_probabilities_tests()
    count_matrix_tests()
    probability_matrix_tests()
    perplexity_tests()
    suggest_a_word_tests()
    get_suggestions_tests()