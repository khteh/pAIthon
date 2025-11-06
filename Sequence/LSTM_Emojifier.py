import argparse, numpy, spacy, tensorflow as tf, csv, emoji
from pathlib import Path
from utils.TrainingMetricsPlot import PlotModelHistory
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Embedding
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from utils.TermColour import bcolors
from utils.GPU import InitializeGPU, SetMemoryLimit
class LSTMEmojifier():
    """
    LSTM model that takes word sequences as input! This model will be able to account for word ordering.
    Emojifier-V2 will continue to use pre-trained word embeddings to represent words. You'll feed word embeddings into an LSTM, and the LSTM will learn to predict the most appropriate emoji.

    Train Keras using mini-batches. However, most deep learning frameworks require that all sequences in the same mini-batch have the same length.

    This is what allows vectorization to work: If you had a 3-word sentence and a 4-word sentence, then the computations needed for them are different (one takes 3 steps of an LSTM, one takes 4 steps) so it's just not possible to do them both at the same time.

    Padding Handles Sequences of Varying Length
    The common solution to handling sequences of different length is to use padding. Specifically:
    Set a maximum sequence length
    Pad all sequences to have the same length.
    Example of Padding:
    Given a maximum sequence length of 20, you could pad every sentence with "0"s so that each input sentence is of length 20.
    Thus, the sentence "I love you" would be represented as  (𝑒𝐼,𝑒𝑙𝑜𝑣𝑒,𝑒𝑦𝑜𝑢,0⃗ ,0⃗ ,…,0⃗ )
    .
    In this example, any sentences longer than 20 words would have to be truncated.
    One way to choose the maximum sequence length is to just pick the length of the longest sentence in the training set.    
    """
    _nlp = None
    _path:str = None
    _model_path:str = None
    _words = None
    _word_to_vec_map = None
    _words_to_index = None
    _index_to_words = None
    _max_len: int = None # Needed for padding
    _classes: int = None
    _model: Model = None
    _X_train = None
    _Y_train = None
    _X_test = None
    _Y_test = None
    _X_train_indices = None
    _Y_train_oh = None
    _X_test_indices = None
    _Y_test_oh = None
    _learning_rate: float = None
    _trained: bool = False
    # The entire set of Emoji codes as defined by the Unicode consortium is supported in addition to a bunch of aliases. By default, only the official list is enabled but doing emoji.emojize(language='alias') enables both the full list and aliases.
    _emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                        "1": ":baseball:",
                        "2": ":smile:",
                        "3": ":disappointed:",
                        "4": ":fork_and_knife:"}
    def __init__(self, path: str = None, model_path:str = None, train:str = None, test:str = None, word_to_vec_map = None, word_to_index = None, max_len:int = None, learning_rate:float = 0.01):
        self._path = path
        self._model_path = model_path
        self._max_len = max_len
        self._words_to_index = word_to_index
        self._learning_rate = learning_rate
        # $ pp spacy download en_core_web_md
        self._nlp = spacy.load("en_core_web_md")
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._saved_model = True
            self._model = load_model(self._model_path) # https://github.com/tensorflow/tensorflow/issues/102475
            self._trained = True
        if word_to_vec_map:
            self._word_to_vec_map = word_to_vec_map
        elif path:
            self._path = path
            self._words = set()
            self._word_to_vec_map = {}
            self._read_glove_vecs()
        else:
            raise RuntimeError("Please provide a word_to vec map or path to load from!")
        self._PrepareData(train, test)

    def sentences_to_indices(self, X):
        """
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
        
        Arguments:
        X -- array of sentences (strings), of shape (m,)
        word_to_index -- a dictionary containing the each word mapped to its index
        max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
        
        Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
        """
        m = X.shape[0]                                   # number of training examples
        # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
        X_indices = numpy.zeros((m, self._max_len))
        for i in range(m):                               # loop over training examples
            # Convert the ith training sentence to lower case and split it into words. You should get a list of words.
            sentence_words = [w.lower() for w in X[i].split()]
            j = 0
            # Loop over the words of sentence_words
            for w in sentence_words:
                # if w exists in the word_to_index dictionary
                if w in self._words_to_index:
                    # Set the (i,j)th entry of X_indices to the index of the correct word.
                    X_indices[i, j] = self._words_to_index[w]
                    # Increment j to j + 1
                    j += 1
        return X_indices

    def pretrained_embedding_layer(self):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
        
        Arguments:
        word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        embedding_layer -- pretrained layer Keras instance
        """
        vocab_size = len(self._words_to_index) + 1              # adding 1 to fit Keras embedding (requirement)
        any_word = next(iter(self._word_to_vec_map.keys()))
        emb_dim = self._word_to_vec_map[any_word].shape[0]    # define dimensionality of your GloVe word vectors (= 50)
        
        # Step 1
        # Initialize the embedding matrix as a numpy array of zeros.
        # See instructions above to choose the correct shape.
        emb_matrix = numpy.zeros((vocab_size, emb_dim))
        
        # Step 2
        # Set each row "idx" of the embedding matrix to be 
        # the word vector representation of the idx'th word of the vocabulary
        for word, idx in self._words_to_index.items():
            emb_matrix[idx, :] = self._word_to_vec_map[word]

        # Step 3
        # Define Keras embedding layer with the correct input and output sizes
        # Make it non-trainable.
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=emb_dim, trainable = False, weights = emb_matrix)

        # Step 4 (already done for you; please do not modify)
        # Build the embedding layer, it is required before setting the weights of the embedding layer. 
        embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
        
        # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
        embedding_layer.set_weights([emb_matrix])
        return embedding_layer
    
    def BuildModel(self, retrain:bool = False):
        """
        Function creating the Emojify-v2 model's graph.
        
        Arguments:
        input_shape -- shape of the input, usually (max_len,)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        model -- a model instance in Keras
        """
        if not self._model or retrain:
            self._trained = False
            # Define sentence_indices as the input of the graph.
            # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
            sentence_indices = Input(shape=(self._max_len,), dtype='int32')
            
            # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
            embedding_layer = self.pretrained_embedding_layer()
            
            # Propagate sentence_indices through your embedding layer
            # (See additional hints in the instructions).
            embeddings = embedding_layer(sentence_indices)
            
            # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
            # The returned output should be a batch of sequences.
            X = LSTM(128, return_sequences=True)(embeddings)
            # Add dropout with a probability of 0.5
            X = Dropout(0.5)(X)
            # Propagate X trough another LSTM layer with 128-dimensional hidden state
            # The returned output should be a single hidden state, not a batch of sequences.
            X = LSTM(128, return_sequences=False)(X)
            # Add dropout with a probability of 0.5
            X = Dropout(0.5)(X)
            # Propagate X through a Dense layer with 5 units
            X = Dense(5, activation="softmax", kernel_regularizer=l2(0.01))(X)
            
            # Create Model instance which converts sentence_indices into X.
            self._model = Model(inputs=sentence_indices, outputs=X)
            self._model.compile(
                    loss=CategoricalCrossentropy(from_logits=False), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
                    optimizer=Adam(learning_rate=self._learning_rate), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                    metrics=['accuracy']
                )
            self._model.summary()
            plot_model(
                self._model,
                to_file="output/LSTM_Emojifier.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
    def Train(self, epochs: int = 50, batch_size:int = 32, retrain:bool = False):
        if not self._trained or retrain:
            if self._model:
                history = self._model.fit(self._X_train_indices, self._Y_train_oh, epochs = epochs, batch_size = batch_size, shuffle=True) # shuffle: Boolean, whether to shuffle the training data before each epoch. This argument is ignored when x is a generator or a tf.data.Dataset.
                PlotModelHistory("LSTM Emojifier", history)
                if self._model_path:
                    self._model.save(self._model_path) #https://github.com/tensorflow/tensorflow/issues/102475
                    print(f"Model saved to {self._model_path}.")
            else:
                print(f"{bcolors.FAIL}Please build the model first by calling BuildModel()!{bcolors.DEFAULT}")
    def Evaluate(self):
        loss, acc = self._model.evaluate(self._X_test_indices, self._Y_test_oh)
        print(f"Test accuracy = {acc}, loss: {loss}")
        # This code allows you to see the mislabelled examples
        print(f"{bcolors.WARNING}Mislabelled emojis:{bcolors.DEFAULT}")
        pred = self._model.predict(self._X_test_indices)
        for i in range(len(self._X_test)):
            num = numpy.argmax(pred[i])
            if(num != self._Y_test[i]):
                print(f"Expected emoji: {self._label_to_emoji(self._Y_test[i])} Prediction: {self._X_test[i] + self._label_to_emoji(num).strip()}")

    def Predict(self, input:str):
        x_test = numpy.array([input])
        X_test_indices = self.sentences_to_indices(x_test)
        print(f"{x_test[0]}:  {self._label_to_emoji(numpy.argmax(self._model.predict(X_test_indices)))}")

    def _PrepareData(self, train:str = None, test:str = None):
        print(f"\n=== {self._PrepareData.__name__} ===")
        if train:
            self._X_train, self._Y_train = self._read_csv(train)
            self._max_len = len(max(self._X_train, key=lambda x: len(x.split())).split())
            print(f"X_train: {self._X_train.shape}, Y_train: {self._Y_train.shape}")
            self._classes = len(numpy.unique(self._Y_train)) # Unique number of emojis
            self._X_train_indices = self.sentences_to_indices(self._X_train)
            print(f"{self._classes} classes: {numpy.unique(self._Y_train)}")
            self._Y_train_oh = self._convert_to_one_hot(self._Y_train)
        if test:
            self._X_test, self._Y_test = self._read_csv(test)
            self._X_test_indices = self.sentences_to_indices(self._X_test)
            self._Y_test_oh = self._convert_to_one_hot(self._Y_test)

    def _convert_to_one_hot(self, data):
        return numpy.eye(self._classes)[data.reshape(-1)]

    def _label_to_emoji(self, label):
        """
        Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
        """
        return emoji.emojize(self._emoji_dictionary[str(label)], language='alias')

    def _read_glove_vecs(self):
        with open(self._path, 'r') as f:
            words = set()
            self._word_to_vec_map = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                self._word_to_vec_map[curr_word] = numpy.array(line[1:], dtype=numpy.float64)
            i = 1
            self._words_to_index = {}
            self._index_to_words = {}
            for w in sorted(words):
                self._words_to_index[w] = i
                self._index_to_words[i] = w
                i = i + 1
    def _read_csv(self, filename):
        phrase = []
        emoji = []
        with open (filename, 'r') as f:
            csvReader = csv.reader(f)
            for row in csvReader:
                phrase.append(row[0])
                emoji.append(row[1])
        X = numpy.asarray(phrase)
        Y = numpy.asarray(emoji, dtype=int)
        return X, Y

def sentences_to_indices_test(retrain:bool):
    print(f"\n=== {sentences_to_indices_test.__name__} ===")
    # Create a word_to_index dictionary
    word_to_index = {}
    for idx, val in enumerate(["i", "like", "learning", "deep", "machine", "love", "smile", '´0.=']):
        word_to_index[val] = idx + 1

    max_len = 4
    # def __init__(self, path: str = None, train:str = None, test:str = None, word_to_vec_map = None, word_to_index = None, max_len:int = None, learning_rate:float = 0.01):
    # https://nlp.stanford.edu/projects/glove/
    nlp = LSTMEmojifier('/usr/src/GloVe/glove.6B.300d.txt', "models/LSTM_Emojifier.keras", None, None, None, word_to_index, max_len)
       
    sentences = numpy.array(["I like deep learning", "deep ´0.= love machine", "machine learning smile", "$"]);
    indexes = nlp.sentences_to_indices(sentences)
    #print(indexes)
    
    assert type(indexes) == numpy.ndarray, "Wrong type. Use np arrays in the function"
    assert indexes.shape == (sentences.shape[0], max_len), "Wrong shape of ouput matrix"
    #assert numpy.allclose(indexes, [[1, 2, 4, 3],
    #                             [4, 8, 6, 5],
    #                             [5, 3, 7, 0],
    #                             [0, 0, 0, 0]]), "Wrong values. Debug with the given examples"
    print(f"{bcolors.OKGREEN}All tests passed!{bcolors.DEFAULT}")

def pretrained_embedding_layer_test(retrain:bool):
    print(f"\n=== {pretrained_embedding_layer_test.__name__} ===")
    # Create a controlled word to vec map
    word_to_vec_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4], 
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0], 
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]
                      }
    # Convert lists to numpy.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = numpy.array(word_to_vec_map[key])
        
    # Create a word_to_index dictionary
    word_to_index = {}
    for idx, val in enumerate(list(word_to_vec_map.keys())):
        word_to_index[val] = idx;
    # def __init__(self, path: str = None, train:str = None, test:str = None, word_to_vec_map = None, word_to_index = None, max_len:int = None, learning_rate:float = 0.01):
    # https://nlp.stanford.edu/projects/glove/
    nlp = LSTMEmojifier('/usr/src/GloVe/glove.6B.300d.txt', "models/LSTM_Emojifier.keras", None, None, word_to_vec_map, word_to_index)
    embedding_layer = nlp.pretrained_embedding_layer()
    
    assert type(embedding_layer) == Embedding, "Wrong type"
    assert embedding_layer.input_dim == len(list(word_to_vec_map.keys())) + 1, "Wrong input shape"
    assert embedding_layer.output_dim == len(word_to_vec_map['a']), "Wrong output shape"
    assert numpy.allclose(embedding_layer.get_weights(), 
                       [[[ 3, 3], [ 3, 3], [ 2, 4], [ 3, 2], [ 3, 4],
                       [-2, 1], [-2, 2], [-1, 2], [-1, 1], [-1, 0],
                       [-2, 0], [-3, 0], [-3, 1], [-3, 2], [ 0, 0]]]), "Wrong vaulues"
    print(f"{bcolors.OKGREEN}All tests passed!{bcolors.DEFAULT}")

def model_tests(retrain:bool):
    print(f"\n=== {model_tests.__name__} ===")
    # def __init__(self, path: str = None, train:str = None, test:str = None, word_to_vec_map = None, word_to_index = None, max_len:int = None, learning_rate:float = 0.01):
    # https://nlp.stanford.edu/projects/glove/
    model = LSTMEmojifier('/usr/src/GloVe/glove.6B.300d.txt', "models/LSTM_Emojifier.keras", 'data/Emojifier/train_emoji.csv', 'data/Emojifier/test_emoji.csv')
    model.BuildModel()
    model.Train(100, 32, retrain)
    model.Evaluate()
    sentences = ["The meal was great!", "I had a tough day!", "The job looks interesting!", "I had a great trip!", "I learnt something new today!"]
    for i in sentences:
        model.Predict(i)

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='LSTM Emojifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()
    InitializeGPU()
    sentences_to_indices_test(args.retrain)
    pretrained_embedding_layer_test(args.retrain)
    model_tests(args.retrain)