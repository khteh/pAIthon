from __future__ import print_function
import argparse, numpy, random, pprint, copy, sys, io
from utils.RNN_utils import *
from Softmax import softmax
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.shakespeare_utils import on_epoch_end, sample
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
"""
Exploding gradients
When gradients are very large, they're called "exploding gradients."
Exploding gradients make the training process more difficult, because the updates may be so large that they "overshoot" the optimal values during back propagation.
Recall that your overall loop structure usually consists of:

forward pass,
cost computation,
backward pass,
parameter update.
Before updating the parameters, you will perform gradient clipping to make sure that your gradients are not "exploding."

Gradient clipping
In the exercise below, you will implement a function clip that takes in a dictionary of gradients and returns a clipped version of gradients, if needed.

There are different ways to clip gradients.
You will use a simple element-wise clipping procedure, in which every element of the gradient vector is clipped to fall between some range [-N, N].
For example, if the N=10
The range is [-10, 10]
If any component of the gradient vector is greater than 10, it is set to 10.
If any component of the gradient vector is less than -10, it is set to -10.
If any components are between -10 and 10, they keep their original values.
"""
class CharacterGenerationRNN():
    _path:str = None
    _data_size: int = None
    _words = None
    _vocab_size: int = None
    _char_to_ix = None
    _ix_to_char = None
    _learning_rate: float = None
    _dinasaur: bool = None
    _shakespeare: bool = None
    _user_input:str = None
    _Tx: int = None # sequence length, number of time-steps (or characters) in one training example
    _stride:int = None # how much the window shifts itself while scanning
    _X = []
    _Y = []

    def __init__(self, path:str, learning_rate:float, dinasaur:bool = False, shakespeare:bool = False, user_input:str = None, tx:int = None, stride:int = None):
        self._path = path
        self._learning_rate = learning_rate
        self._dinasaur = dinasaur
        self._shakespeare = shakespeare
        self._user_input = user_input
        self._Tx = tx
        self._stride = stride
        self._PrepareData()

    def _PrepareData(self):
        if self._dinasaur:
            with open(self._path, 'r', newline='') as f: # dinos.txt
                data = f.read().lower()
                chars = list(set(data))
                chars = sorted(chars)
                self._words = data.split("\n")
                self._data_size, self._vocab_size = len(data), len(chars)
                self._char_to_ix = { ch:i for i,ch in enumerate(chars) }
                self._ix_to_char = { i:ch for i,ch in enumerate(chars) }
            print('There are %d total characters and %d unique characters in your data.' % (self._data_size, self._vocab_size))
        elif self._shakespeare:
            print("Loading text data...")
            with open(self._path, encoding='utf-8') as f:
                text = f.read().lower()
                #print('corpus length:', len(text))
                chars = sorted(list(set(text)))
                self._vocab_size = len(chars)
                self._char_to_ix = { ch:i for i,ch in enumerate(chars) }
                self._ix_to_char = { i:ch for i,ch in enumerate(chars) }
                #print('number of unique characters in the corpus:', len(chars))
                print("Creating training set...")
                for i in range(0, len(text) - self._Tx, self._stride):
                    self._X.append(text[i: i + self._Tx])
                    self._Y.append(text[i + self._Tx])
                print('number of training examples:', len(self._X))
            print("Vectorizing training set...")
            x, y = self._vectorization()

    def VocabSize(self) -> int:
        return self._vocab_size
    def ix_to_char(self, idx:int):
        return self._ix_to_char[idx]
    def char_to_ix(self, char:int):
        return self._char_to_ix[char]
    def IndexSize(self):
        return len(self._char_to_ix)
    def _vectorization(self):
        """
        Convert X and Y (lists) into arrays to be given to a recurrent neural network.
        
        Arguments:
        X -- 
        Y -- 
        Tx -- integer, sequence length
        
        Returns:
        x -- array of shape (m, Tx, len(chars))
        y -- array of shape (m, len(chars))
        """
        m = len(self._X)
        x = numpy.zeros((m, self._Tx, self._vocab_size), dtype=numpy.bool)
        y = numpy.zeros((m, self._vocab_size), dtype=numpy.bool)
        for i, sentence in enumerate(self._X):
            for t, char in enumerate(sentence):
                x[i, t, self._char_to_ix[char]] = 1
            y[i, self._char_to_ix[self._Y[i]]] = 1
        return x, y 
    
    def Clip(self, gradients, maxValue):
        '''
        Clips the gradients' values between minimum and maximum.
        
        Arguments:
        gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
        
        Returns: 
        gradients -- a dictionary with the clipped gradients.
        '''
        gradients = copy.deepcopy(gradients)
        dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
        # Clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
        for gradient in gradients:
            numpy.clip(gradients[gradient], -maxValue, maxValue, out = gradients[gradient])
        gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
        return gradients

    def Sample(self, parameters, seed=0):
        """
        Sample a sequence of characters according to a sequence of probability distributions output of the RNN

        Arguments:
        parameters -- Python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
        char_to_ix -- Python dictionary mapping each character to an index.
        seed -- Used for grading purposes. Do not worry about it.

        Returns:
        indices -- A list of length n containing the indices of the sampled characters.
        """
        # Retrieve parameters and relevant shapes from "parameters" dictionary
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
        vocab_size = by.shape[0]
        n_a = Waa.shape[1]
        print(f"Waa: {Waa.shape}, Wax: {Wax.shape}, Wya: {Wya.shape}, by: {by.shape}, b: {b.shape}")
        
        ### START CODE HERE ###
        # Step 1: Create the a zero vector x that can be used as the one-hot vector 
        # Representing the first character (initializing the sequence generation). (≈1 line)
        x = numpy.zeros((vocab_size,1))
        # Step 1': Initialize a_prev as zeros (≈1 line)
        a_prev = numpy.zeros((n_a,1))
        print(f"Wax: {Wax.shape}, x: {x.shape}, Waa: {Waa.shape}, a_prev: {a_prev.shape}, n_a: {n_a}")
        # Create an empty list of indices. This is the list which will contain the list of indices of the characters to generate (≈1 line)
        indices = []
        
        # idx is the index of the one-hot vector x that is set to 1
        # All other positions in x are zero.
        # Initialize idx to -1
        idx = -1
        
        # Loop over time-steps t. At each time-step:
        # Sample a character from a probability distribution 
        # And append its index (`idx`) to the list "indices". 
        # You'll stop if you reach 50 characters 
        # (which should be very unlikely with a well-trained model).
        # Setting the maximum number of characters helps with debugging and prevents infinite loops. 
        counter = 0
        newline_character = self._char_to_ix['\n']
        
        while (idx != newline_character and counter != 50):
            
            # Step 2: Forward propagate x using the equations (1), (2) and (3)
            # g(Wa @ [a(t-1), x(t)] + ba)
            assert (n_a, vocab_size) == Wax.shape
            assert (vocab_size, 1) == x.shape
            a = numpy.tanh(Wax @ x + Waa @ a_prev + b) # Similar to sigmoid graph but the output is [-1, 1]
            assert (n_a,1) == a.shape
            z = Wya @ a + by
            assert (vocab_size, 1) == z.shape
            y = softmax(z)
            assert (vocab_size, 1) == y.shape
            #print(f"a: {a.shape}, z: {z.shape}, y: {y.shape}, {y}")
            # For grading purposes
            numpy.random.seed(counter + seed) 
            
            # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
            # (see additional hints above)
            probs = y.ravel()
            #print(f"y: {y.shape}, {y[0]} sum: {numpy.sum(y)}, probs: {probs.shape}, {probs}")
            idx = rng.choice(range(len(probs)), p = probs)

            # Append the index to "indices"
            indices.append(idx)
            
            # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
            # (see additional hints above)
            x = numpy.zeros((vocab_size,1))
            x[idx] = 1
            
            # Update "a_prev" to be "a"
            a_prev = a
            
            # for grading purposes
            seed += 1
            counter +=1
            
        if (counter == 50):
            indices.append(self._char_to_ix['\n'])
        
        return indices

    def Optimize(self, X, Y, a_prev, parameters, learning_rate = 0.01):
        """
        Execute one step of the optimization (stochastic gradient descent (with clipped gradients)) to train the model.
        Since it goes through the training examples one at a time, the optimization algorithm will be stochastic gradient descent.

        As a reminder, here are the steps of a common optimization loop for an RNN:
        (1) Forward propagate through the RNN to compute the loss
        (2) Backward propagate through time to compute the gradients of the loss with respect to the parameters
        (3) Clip the gradients
        (4) Update the parameters using gradient descent

        Arguments:
        X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
        Y -- list of integers, exactly the same as X but shifted one index to the left.
        a_prev -- previous hidden state.
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            b --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        learning_rate -- learning rate for the model.
        
        Returns:
        loss -- value of the loss function (cross-entropy)
        gradients -- python dictionary containing:
                            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                            dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                            db -- Gradients of bias vector, of shape (n_a, 1)
                            dby -- Gradients of output bias vector, of shape (n_y, 1)
        a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
        """
        
        # Forward propagate through time (≈1 line)
        loss, cache = rnn_forward(X, Y, a_prev, parameters)
        
        # Backpropagate through time (≈1 line)
        gradients, a = rnn_backward(X, Y, parameters, cache)
        
        # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
        gradients = self.Clip(gradients, 5)
        
        # Update parameters (≈1 line)
        parameters = update_parameters(parameters, gradients, learning_rate)
        return loss, gradients, a[len(X)-1]
    
    def BuildTrainShakespeareModel(self):
        if not self._shakespeare:
            print(f"Please select shakepeare poen generation functionality with -s option on the command line")
            return
        if not self._user_input or len(self._user_input) <= 0:
            self._user_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
            if not self._user_input or len(self._user_input) <= 0:
                print(f"Cannot continue without your input")
                return
        print("Loading model...")
        model = load_model('models/model_shakespeare_kiank_350_epoch.h5')
        #model = load_model('models/model_shakespeare_kiank.h5')
        generated = ''
        #sentence = text[start_index: start_index + Tx]
        #sentence = '0'*Tx
        #usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
        # zero pad the sentence to Tx characters.
        sentence = ('{0:0>' + str(self._Tx) + '}').format(self._user_input).lower()
        generated += self._user_input 

        sys.stdout.write("\n\nHere is your poem: \n\n") 
        sys.stdout.write(self._user_input )
        for i in range(400):

            x_pred = numpy.zeros((1, self._Tx, len(self._vocab_size)))

            for t, char in enumerate(sentence):
                if char != '0':
                    x_pred[0, t, self._char_to_ix[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature = 1.0)
            next_char = self._ix_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

            if next_char == '\n':
                continue

    def BuildTrainDinasaurModel(self, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
        """
        Trains the model and generates dinosaur names. 
        
        Arguments:
        data_x -- text corpus, divided in words
        ix_to_char -- dictionary that maps the index to a character
        char_to_ix -- dictionary that maps a character to an index
        num_iterations -- number of iterations to train the model for
        n_a -- number of units of the RNN cell
        dino_names -- number of dinosaur names you want to sample at each iteration. 
        vocab_size -- number of unique characters found in the text (size of the vocabulary)
        
        Returns:
        parameters -- learned parameters
        """
        
        # Retrieve n_x and n_y from vocab_size
        n_x, n_y = vocab_size, vocab_size
        
        # Initialize parameters
        parameters = initialize_parameters(n_a, n_x, n_y)
        
        # Initialize loss (this is required because we want to smooth our loss)
        loss = get_initial_loss(vocab_size, dino_names)
        
        # Build list of all dinosaur names (training examples).
        examples = [x.strip() for x in self._words]
        #print(f"examples: {examples}")
        # Shuffle list of all dinosaur names
        #numpy.random.seed(0)
        numpy.random.shuffle(examples)
        
        # Initialize the hidden state of your LSTM
        a_prev = numpy.zeros((n_a, 1))
        
        # for grading purposes
        last_dino_name = "abc"
        
        # Optimization loop
        for j in range(num_iterations):
            
            ### START CODE HERE ###
            
            # Set the index `idx` (see instructions above)
            idx = j % len(examples)
            #print(f"idx: {idx}")
            # Set the input X (see instructions above)
            single_example = examples[idx]
            #print(f"examples[{idx}]: {single_example}")
            single_example_chars = [c for c in single_example]
            single_example_ix = [self._char_to_ix[c] for c in single_example_chars]
            X = [None] + single_example_ix
            
            # Set the labels Y (see instructions above)
            ix_newline = self._char_to_ix['\n']
            Y = X[1:] + [ix_newline]

            # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
            # Choose a learning rate of 0.01
            curr_loss, gradients, a_prev = self.Optimize(X, Y, a_prev, parameters, 0.01)
            
            ### END CODE HERE ###
            
            # debug statements to aid in correctly forming X, Y
            if verbose and j in [0, len(examples) -1, len(examples)]:
                print("j = " , j, "idx = ", idx,) 
            if verbose and j in [0]:
                print("single_example =", single_example)
                print("single_example_chars", single_example_chars)
                print("single_example_ix", single_example_ix)
                print(" X = ", X, "\n", "Y =       ", Y, "\n")
            
            # to keep the loss smooth.
            loss = smooth(loss, curr_loss)

            # Every 2000 Iteration, generate "n" characters thanks to self.Sample() to check if the model is learning properly
            if j % 2000 == 0:
                print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
                # The number of dinosaur names to print
                seed = 0
                for name in range(dino_names):
                    # Sample indices and print them
                    sampled_indices = self.Sample(parameters, seed)
                    last_dino_name = get_sample(sampled_indices, self._ix_to_char)
                    print(last_dino_name.replace('\n', ''))
                    seed += 1  # To get the same result (for grading purposes), increment the seed by one. 
                print('\n')
        return parameters, last_dino_name
    
    def GenerateShakespearePoem(self):
        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

# Test with a max value of 10
def clip_test(mValue):
    chargen = CharacterGenerationRNN("data/dinos.txt", 0.01, args.dinasaur, False)
    print(f"\nGradients for mValue={mValue}")
    numpy.random.seed(3)
    dWax = rng.standard_normal((5, 3)) * 10
    dWaa = rng.standard_normal((5, 5)) * 10
    dWya = rng.standard_normal((2, 5)) * 10
    db = rng.standard_normal((5, 1)) * 10
    dby = rng.standard_normal((2, 1)) * 10
    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}

    gradients2 = chargen.Clip(gradients, mValue)
    print("gradients[\"dWaa\"][1][2] =", gradients2["dWaa"][1][2])
    print("gradients[\"dWax\"][3][1] =", gradients2["dWax"][3][1])
    print("gradients[\"dWya\"][1][2] =", gradients2["dWya"][1][2])
    print("gradients[\"db\"][4] =", gradients2["db"][4])
    print("gradients[\"dby\"][1] =", gradients2["dby"][1])
    
    for grad in gradients2.keys():
        valuei = gradients[grad]
        valuef = gradients2[grad]
        mink = numpy.min(valuef)
        maxk = numpy.max(valuef)
        assert mink >= -abs(mValue), f"Problem with {grad}. Set a_min to -mValue in the numpy.clip call"
        assert maxk <= abs(mValue), f"Problem with {grad}.Set a_max to mValue in the numpy.clip call"
        index_not_clipped = numpy.logical_and(valuei <= mValue, valuei >= -mValue)
        assert numpy.all(valuei[index_not_clipped] == valuef[index_not_clipped]), f" Problem with {grad}. Some values that should not have changed, changed during the clipping process."
    print("\033[92mAll tests passed!\x1b[0m")

def sample_test():
    chargen = CharacterGenerationRNN("data/dinos.txt", 0.01, args.dinasaur, False)
    numpy.random.seed(24)
    _, n_a = 20, 100
    Wax, Waa, Wya = rng.standard_normal((n_a, chargen.VocabSize())), rng.standard_normal((n_a, n_a)), rng.standard_normal((chargen.VocabSize(), n_a))
    b, by = rng.standard_normal((n_a, 1)), rng.standard_normal((chargen.VocabSize(), 1))
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    indices = chargen.Sample(parameters, 0)
    print("Sampling:")
    print(f"list of sampled indices ({len(indices)}): {' '.join(str(i) for i in indices)}")
    print(f"list of sampled characters: {[chargen.ix_to_char(i) for i in indices]}")
    
    assert len(indices) < 52, "Indices length must be smaller than 52"
    assert indices[-1] == chargen.char_to_ix('\n'), "All samples must end with \\n"
    assert min(indices) >= 0 and max(indices) < chargen.IndexSize(), f"Sampled indexes must be between 0 and len(char_to_ix)={chargen.IndexSize()}"
    #assert numpy.allclose(indices, [23, 16, 26, 26, 24, 3, 21, 1, 7, 24, 15, 3, 25, 20, 6, 13, 10, 8, 20, 12, 2, 0]), "Wrong values"
    print("\033[92mAll tests passed!")

def optimize_test():
    chargen = CharacterGenerationRNN("data/dinos.txt", 0.01, args.dinasaur, False)
    numpy.random.seed(1)
    vocab_size, n_a = 27, 100
    a_prev = rng.standard_normal((n_a, 1))
    Wax, Waa, Wya = rng.standard_normal((n_a, vocab_size)), rng.standard_normal((n_a, n_a)), rng.standard_normal((vocab_size, n_a))
    b, by = rng.standard_normal((n_a, 1)), rng.standard_normal((vocab_size, 1))
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    X = [12, 3, 5, 11, 22, 3]
    Y = [4, 14, 11, 22, 25, 26]
    old_parameters = copy.deepcopy(parameters)
    loss, gradients, a_last = chargen.Optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
    print("Loss =", loss)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("numpy.argmax(gradients[\"dWax\"]) =", numpy.argmax(gradients["dWax"]))
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"db\"][4] =", gradients["db"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])
    print("a_last[4] =", a_last[4])
    
    assert numpy.isclose(loss, 126.5039757), "Problems with the call of the rnn_forward function"
    for grad in gradients.values():
        assert numpy.min(grad) >= -5, "Problems in the clip function call"
        assert numpy.max(grad) <= 5, "Problems in the clip function call"
    assert numpy.allclose(gradients['dWaa'][1, 2], 0.1947093), "Unexpected gradients. Check the rnn_backward call"
    assert numpy.allclose(gradients['dWya'][1, 2], -0.007773876), "Unexpected gradients. Check the rnn_backward call"
    assert not numpy.allclose(parameters['Wya'], old_parameters['Wya']), "parameters were not updated"   
    print("\033[92mAll tests passed!")

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Character-level text generation using RNN')
    parser.add_argument('-d', '--dinasaur', action='store_true', help='Generates a cool dinasaur name')
    parser.add_argument('-s', '--shakespeare', action='store_true', help='Generates a shakespeare poem based on your initial input text')
    args = parser.parse_args()
    if args.dinasaur:
        chargen = CharacterGenerationRNN("data/dinos.txt", 0.01, args.dinasaur, False)
        clip_test(10)
        clip_test(5)
        sample_test()
        parameters, last_name = chargen.BuildTrainDinasaurModel(22001, verbose = True)
        print(f"Generated dinasaur name: {last_name}")
    elif args.shakespeare:
        user_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
        chargen = CharacterGenerationRNN("data/shakespeare.txt", 0.01, False, args.shakespeare, user_input, 40, 3)
        chargen.BuildTrainShakespeareModel()
    else:
        print(f"Please select what do you want to generate: -d for dinasaur name, -s for shakespeare's poem")