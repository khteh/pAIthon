import numpy, spacy, csv, emoji
from tqdm import tqdm
from Softmax import softmax
from utils.ConfusionMatrix import ConfusionMatrix
class Emojifier():
    _nlp = None
    _path:str = None
    _words = None
    _word_to_vec_map = None
    _emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                        "1": ":baseball:",
                        "2": ":smile:",
                        "3": ":disappointed:",
                        "4": ":fork_and_knife:"}

    def __init__(self, path: str = None, word_to_vec_map = None):
        # $ pp spacy download en_core_web_md
        self._nlp = spacy.load("en_core_web_md")
        if word_to_vec_map:
            self._word_to_vec_map = word_to_vec_map
        elif path:
            self._path = path
            self._words = set()
            self._word_to_vec_map = {}
            self._read_glove_vecs()
        else:
            raise RuntimeError("Please provide a word_to vec map or path to load from!")

    def sentence_to_avg(self, sentence):
        """
        Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
        and averages its value into a single vector encoding the meaning of the sentence.
        
        Arguments:
        sentence -- string, one training example from X
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        
        Returns:
        avg -- average vector encoding information about the sentence, numpy-array of shape (J,), where J can be any number
        """
        # Get a valid word contained in the word_to_vec_map. 
        any_word = next(iter(self._word_to_vec_map.keys()))
        
        # Step 1: Split sentence into list of lower case words (≈ 1 line)
        words = [w.lower() for w in sentence.split()]

        # Initialize the average word vector, should have the same shape as your word vectors.
        # Use `numpy.zeros` and pass in the argument of any word's word 2 vec's shape
        avg = numpy.zeros(self._word_to_vec_map[any_word].shape)
        
        # Initialize count to 0
        count = 0
        
        # Step 2: average the word vectors. You can loop over the words in the list "words".
        for w in words:
            # Check that word exists in word_to_vec_map
            if w in self._word_to_vec_map:
                avg += self._word_to_vec_map[w]
                # Increment count
                count +=1
           
        if count > 0:
            # Get the average. But only if count > 0
            avg /= count
        return avg

    def BuildModel(self, X, Y, learning_rate = 0.01, num_iterations = 400):
        """
        Model to train word vector representations in numpy.
        
        Arguments:
        X -- input data, numpy array of sentences as strings, of shape (m,)
        Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        learning_rate -- learning_rate for the stochastic gradient descent algorithm
        num_iterations -- number of iterations
        
        Returns:
        pred -- vector of predictions, numpy-array of shape (m, 1)
        W -- weight matrix of the softmax layer, of shape (n_y, n_h)
        b -- bias of the softmax layer, of shape (n_y,)
        """
        
        # Get a valid word contained in the word_to_vec_map 
        any_word = next(iter(self._word_to_vec_map.keys()))
            
        # Define number of training examples
        m = Y.shape[0]                             # number of training examples
        n_y = len(numpy.unique(Y))                    # number of classes  
        n_h = self._word_to_vec_map[any_word].shape[0]   # dimensions of the GloVe vectors 
        
        # Initialize parameters using Xavier initialization
        W = numpy.random.randn(n_y, n_h) / numpy.sqrt(n_h)
        b = numpy.zeros((n_y,))
        
        # Convert Y to Y_onehot with n_y classes
        Y_oh = self._convert_to_one_hot(Y, C = n_y) 
        
        # Optimization loop
        for t in range(num_iterations): # Loop over the number of iterations
            
            cost = 0
            dW = 0
            db = 0
            
            for i in range(m):          # Loop over the training examples
                
                ### START CODE HERE ### (≈ 4 lines of code)
                # Average the word vectors of the words from the i'th training example
                # Use 'sentence_to_avg' you implemented above for this  
                avg = self.sentence_to_avg(X[i])

                # Forward propagate the avg through the softmax layer. 
                # You can use numpy.dot() to perform the multiplication.
                z = W @ avg + b
                a = softmax(z)

                # Add the cost using the i'th training label's one hot representation and "A" (the output of the softmax)
                cost += Y_oh[i] @ numpy.log(a)
                ### END CODE HERE ###
                
                # Compute gradients 
                dz = a - Y_oh[i]
                dW += numpy.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
                db += dz

                # Update parameters with Stochastic Gradient Descent
                W = W - learning_rate * dW
                b = b - learning_rate * db
                
            assert type(cost) == numpy.float64, "Incorrect implementation of cost"
            assert cost.shape == (), "Incorrect implementation of cost"
            
            if t % 100 == 0:
                print("Epoch: " + str(t) + " --- cost = " + str(cost))
                pred = self.Predict(X, Y, W, b) #predict is defined in emo_utils.py
        return pred, W, b
    
    def Predict(self, X, Y, W, b):
        """
        Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.
        
        Arguments:
        X -- input data containing sentences, numpy array of shape (m, None)
        Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)
        
        Returns:
        pred -- numpy array of shape (m, 1) with your predictions
        """
        m = X.shape[0]
        pred = numpy.zeros((m, 1))
        any_word = list(self._word_to_vec_map.keys())[0]
        # number of classes  
        n_h = self._word_to_vec_map[any_word].shape[0] 
        
        for j in range(m):                       # Loop over training examples
            
            # Split jth test example (sentence) into list of lower case words
            words = X[j].lower().split()
            
            # Average words' vectors
            avg = numpy.zeros((n_h,))
            count = 0
            for w in words:
                if w in self._word_to_vec_map:
                    avg += self._word_to_vec_map[w]
                    count += 1
            
            if count > 0:
                avg = avg / count

            # Forward propagation
            Z = numpy.dot(W, avg) + b
            A = softmax(Z)
            pred[j] = numpy.argmax(A)
        print("Accuracy: "  + str(numpy.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
        return pred
    def _label_to_emoji(self, label):
        """
        Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
        """
        return emoji.emojize(self._emoji_dictionary[str(label)], language='alias')
    
    def print_predictions(self, X, pred):
        print()
        for i in range(X.shape[0]):
            print(X[i], self._label_to_emoji(int(pred[i,0])))
    
    def _convert_to_one_hot(self, data, C):
        return numpy.eye(C)[data.reshape(-1)]
    
    def read_csv(self, filename):
        phrase = []
        emoji = []

        with open (filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)

            for row in csvReader:
                phrase.append(row[0])
                emoji.append(row[1])

        X = numpy.asarray(phrase)
        Y = numpy.asarray(emoji, dtype=int)

        return X, Y
    
    def _read_glove_vecs(self):
        with open(self._path, 'r') as f:
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                self._words.add(curr_word)
                self._word_to_vec_map[curr_word] = numpy.array(line[1:], dtype=numpy.float64)

def sentence_to_avg_tests():
    nlp = Emojifier('data/glove.6B.50d.txt')
    avg = nlp.sentence_to_avg("Morrocan couscous is my favorite dish")
    print("avg = \n", avg)
    # Create a controlled word to vec map
    word_to_vec_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0], 
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]
                      }
    # Convert lists to numpy.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = numpy.array(word_to_vec_map[key])
    nlp = Emojifier(None, word_to_vec_map)
    avg = nlp.sentence_to_avg("a a_nw c_w a_s")
    assert tuple(avg.shape) == tuple(word_to_vec_map['a'].shape),  "Check the shape of your avg array"  
    assert numpy.allclose(avg, [1.25, 2.5]),  "Check that you are finding the 4 words"
    avg = nlp.sentence_to_avg("love a a_nw c_w a_s")
    assert numpy.allclose(avg, [1.25, 2.5]), "Divide by count, not len(words)"
    avg = nlp.sentence_to_avg("love")
    assert numpy.array_equal(avg, [0, 0]), "Average of no words must give an array of zeros"
    avg = nlp.sentence_to_avg("c_se foo a a_nw c_w a_s deeplearning c_nw")
    assert numpy.allclose(avg, [0.1666667, 2.0]), "Debug the last example"
    print("\033[92mAll tests passed!")

def model_tests():
    # Create a controlled word to vec map
    word_to_vec_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4], 
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0], 
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]
                      }
    # Convert lists to numpy.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = numpy.array(word_to_vec_map[key])
    # Training set. Sentences composed of a_* words will be of class 0 and sentences composed of c_* words will be of class 1
    X = numpy.asarray(['a a_s synonym_of_a a_n c_sw', 'a a_s a_n c_sw', 'a_s  a a_n', 'synonym_of_a a a_s a_n c_sw', " a_s a_n",
                    " a a_s a_n c ", " a_n  a c c c_e missing",
                   'c c_nw c_n c c_ne', 'c_e c c_se c_s', 'c_nw c a_s c_e c_e', 'c_e a_nw c_sw', 'c_sw c c_ne c_ne'])
    
    Y = numpy.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    nlp = Emojifier(None, word_to_vec_map)
    # def BuildModel(self, X, Y, learning_rate = 0.01, num_iterations = 400):
    pred, W, b = nlp.BuildModel(X, Y, 0.0025, 110)
   
    assert W.shape == (2, 2), "W must be of shape 2 x 2"
    assert numpy.allclose(pred.transpose(), Y), "Model must give a perfect accuracy"
    assert numpy.allclose(b[0], -1 * b[1]), "b should be symmetric in this example"

    X_train, Y_train = nlp.read_csv('data/train_emoji.csv')
    X_test, Y_test = nlp.read_csv('data/tesss.csv')

    nlp = Emojifier('data/glove.6B.50d.txt')
    pred, W, b = nlp.BuildModel(X_train, Y_train)
    #print(f"Y_train prediction: {pred}")
    print("Training set:")
    pred_train = nlp.Predict(X_train, Y_train, W, b)
    print('Test set:')
    pred_test = nlp.Predict(X_test, Y_test, W, b)

    X_my_sentences = numpy.array(["i treasure you", "i love you", "funny lol", "lets play with a ball", "food is ready", "today is not good"])
    Y_my_labels = numpy.array([[0], [0], [2], [1], [4],[3]])

    pred = nlp.Predict(X_my_sentences, Y_my_labels , W, b)
    nlp.print_predictions(X_my_sentences, pred)
    print(f"Y_test: {Y_test.shape}")
    ConfusionMatrix(Y_test.reshape(1,56), pred_test.reshape(1,56), "Emojifier") #assert truths.shape[0] == 1
    print("\033[92mAll tests passed!")

if __name__ == "__main__":
    sentence_to_avg_tests()
    model_tests()