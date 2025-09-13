import tensorflow.keras.backend as K
import numpy, tensorflow as tf, random, matplotlib.pyplot as plt
from faker import Faker
from tqdm import tqdm
from babel.dates import format_date
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model

class MachineTranslation():
    # Define format of the data we would like to generate
    FORMATS = ['short',
            'medium',
            'long',
            'full',
            'full',
            'full',
            'full',
            'full',
            'full',
            'full',
            'full',
            'full',
            'full',
            'd MMM YYY', 
            'd MMMM YYY',
            'dd MMM YYY',
            'd MMM, YYY',
            'd MMMM, YYY',
            'dd, MMM YYY',
            'd MM YY',
            'd MMMM YYY',
            'MMMM d YYY',
            'MMMM d, YYY',
            'dd.MM.YY']
    # change this if you want it to work with another language
    LOCALES = ['en_SG']
    _fake: None
    _size:int = None
    _locale:str = None
    _Tx: int = None
    _Ty: int = None
    _dataset = None
    _human_vocab = None
    _machine_vocab = None
    _inv_machine_vocab = None
    _X = None
    _Y = None
    _Xoh = None
    _Yoh = None
    _n_a:int = None# -- hidden state size of the Bi-LSTM
    _n_s:int = None# -- hidden state size of the post-attention LSTM
    _model: Model = None

    def __init__(self, locale:str, size:int, tx:int, ty:int, n_a:int, n_s:int):
        self._locale = locale
        self._size = size
        self._Tx = tx
        self._Ty = ty
        self._n_a = n_a
        self._n_s = n_s
        self._fake = Faker()
        self._PrepareData()

    def one_step_attention(self, a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        
        Returns:
        context -- context vector, input of the next (post-attention) LSTM cell
        """
        print(f"\n=== {self.one_step_attention.__name__} ===")
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = RepeatVector(self._Tx)(s_prev)
        print(f"a.shape: {a.shape}, s_prev.shape: {s_prev.shape}")
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        # For grading purposes, please list 'a' first and 's_prev' second, in this order.
        concat = Concatenate(axis=-1)([a, s_prev])
        #print(f"a: {a.shape}, concat: {concat.shape}")
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = Dense(10, activation='tanh')(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = Dense(1, activation='relu')(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = Activation(self._softmax, name='attention_weights')(energies) # We are using a custom softmax(axis = 1) loaded in this notebook
        # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = Dot(axes=1)([alphas, a])
        #print(f"context: {context.shape} {context.numpy()}")
        return context

    def BuildModel(self):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- integer, optional, size of the python dictionary "machine_vocab"
                            This is not being used

        Returns:
        model -- Keras model instance
        """
        print(f"\n=== {self.BuildModel.__name__} ===")
        # Define the inputs of your model with a shape (Tx, human_vocab_size)
        # Define s0 (initial hidden state) and c0 (initial cell state)
        # for the decoder LSTM with shape (n_s,)
        X = Input(shape=(self._Tx, len(self._human_vocab)))
        # initial hidden state
        s0 = Input(shape=(self._n_s,), name='s0')
        # initial cell state
        c0 = Input(shape=(self._n_s,), name='c0')
        # hidden state
        s = s0
        # cell state
        c = c0
        
        # Initialize empty list of outputs
        outputs = []
        
        # Step 1: Define your pre-attention Bi-LSTM. (≈ 1 line)
        a = Bidirectional(LSTM(units=self._n_a, return_sequences=True))(X)

        # Step 2: Iterate for Ty steps
        for t in range(self._Ty):
        
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = self.one_step_attention(a, s)
            
            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector. (≈ 1 line)
            # Don't forget to pass: initial_state = [hidden state, cell state] 
            # Remember: s = hidden state, c = cell state
            # Remember to pass in the previous hidden-state  𝑠⟨𝑡−1⟩ and cell-states  𝑐⟨𝑡−1⟩ of this LSTM
            print(f"s: {s.shape}, c: {c.shape}, context: {context.shape}")
            _, s, c = LSTM(self._n_s, return_state = True, return_sequences=True)(inputs=context, initial_state=[s, c])
            
            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = Dense(len(self._machine_vocab), activation=softmax)(s)
            
            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)
        
        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        self._model = Model(inputs=[X,s0,c0], outputs=outputs)
        """
        expected_summary = [['InputLayer', [(None, 30, 37)], 0],
                         ['InputLayer', [(None, 64)], 0],
                         ['Bidirectional', (None, 30, 64), 17920],
                         ['RepeatVector', (None, 30, 64), 0, 30],
                         ['Concatenate', (None, 30, 128), 0],
                         ['Dense', (None, 30, 10), 1290, 'tanh'],
                         ['Dense', (None, 30, 1), 11, 'relu'],
                         ['Activation', (None, 30, 1), 0],
                         ['Dot', (None, 1, 64), 0],
                         ['InputLayer', [(None, 64)], 0],
                         ['LSTM',[(None, 64), (None, 64), (None, 64)], 33024,[(None, 1, 64), (None, 64), (None, 64)],'tanh'],
                         ['Dense', (None, 11), 715, 'softmax']]
        """
        assert len(self._model.outputs) == 10, f"Wrong output shape. Expected 10 != {len(self._model.outputs)}"
        self._model.summary()

    def _softmax(self, x, axis=1):
        """Softmax activation function.
        # Arguments
            x : Tensor.
            axis: Integer, axis along which the softmax normalization is applied.
        # Returns
            Tensor, output of softmax transformation.
        # Raises
            ValueError: In case `dim(x) == 1`.
        """
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')

    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        """
        _size:int = None
        _locale:str = None
        _Tx: int = None
        _Ty: int = None
        _dataset = None
        _human_vocab = None
        _machine_vocab = None
        _inv_machine_vocab = None
        _X = None
        _Y = None
        _Xoh = None
        _Yoh = None
        _n_a:int = None# -- hidden state size of the Bi-LSTM
        _n_s:int = None# -- hidden state size of the post-attention LSTM
        """
        self._load_dataset()
        #Tx = 30
        #Ty = 10
        self._preprocess_data()
        print(f"size: {self._size}, Tx: {self._Tx}, Ty: {self._Ty}, n_a: {self._n_a}, n_s: {self._n_s}, X: {self._X.shape}, Y: {self._Y.shape}, Xoh: {self._Xoh.shape}, Yoh: {self._Yoh.shape}")
        index = 0
        print(f"Source date: {self._dataset[index][0]}")
        print(f"Target date: {self._dataset[index][1]}")
        print(f"\nSource after preprocessing (indices): {self._X[index]}")
        print(f"Target after preprocessing (indices): {self._Y[index]}")
        print(f"\nSource after preprocessing (one-hot) {self._Xoh[index]}")
        print(f"Target after preprocessing (one-hot): {self._Yoh[index]}")

    def _load_date(self):
        """
            Loads some fake dates 
            :returns: tuple containing human readable string, machine readable string, and date object
        """
        dt = self._fake.date_object()
        try:
            human_readable = format_date(dt, format=random.choice(self.FORMATS),  locale=self._locale) # locale=random.choice(LOCALES))
            human_readable = human_readable.lower()
            human_readable = human_readable.replace(',','')
            machine_readable = dt.isoformat()
        except AttributeError as e:
            return None, None, None
        return human_readable, machine_readable, dt

    def _load_dataset(self):
        """
            Loads a dataset with m examples and vocabularies
            :m: the number of examples to generate
        """
        human_vocab = set()
        machine_vocab = set()
        self._dataset = []
        #Tx = 30
        for i in tqdm(range(self._size)):
            h, m, _ = self._load_date()
            if h is not None:
                self._dataset.append((h, m))
                human_vocab.update(tuple(h))
                machine_vocab.update(tuple(m))
        self._human_vocab = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], list(range(len(human_vocab) + 2))))
        self._inv_machine_vocab = dict(enumerate(sorted(machine_vocab)))
        self._machine_vocab = {v:k for k,v in self._inv_machine_vocab.items()}

    def _preprocess_data(self):
        self._X, self._Y = zip(*self._dataset)
        self._X = numpy.array([self._string_to_int(i, self._Tx, self._human_vocab) for i in self._X])
        self._Y = numpy.array([self._string_to_int(t, self._Ty, self._machine_vocab) for t in self._Y])
        self._Xoh = numpy.array(list(map(lambda x: to_categorical(x, num_classes=len(self._human_vocab)), self._X)))
        self._Yoh = numpy.array(list(map(lambda x: to_categorical(x, num_classes=len(self._machine_vocab)), self._Y)))

    def _string_to_int(self, string, length, vocab):
        """
        Converts all strings in the vocabulary into a list of integers representing the positions of the
        input string's characters in the "vocab"
        
        Arguments:
        string -- input string, e.g. 'Wed 10 Jul 2007'
        length -- the number of time steps you'd like, determines if the output will be padded or cut
        vocab -- vocabulary, dictionary used to index every character of your "string"
        
        Returns:
        rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
        """
        #make lower to standardize
        string = string.lower()
        string = string.replace(',','')
        if len(string) > length:
            string = string[:length]
        rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
        if len(string) < length:
            rep += [vocab['<pad>']] * (length - len(string))
        #print (rep)
        return rep

    def _int_to_string(self, ints, inv_vocab):
        """
        Output a machine readable list of characters based on a list of indexes in the machine's vocabulary
        
        Arguments:
        ints -- list of integers representing indexes in the machine's vocabulary
        inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 
        
        Returns:
        l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
        """
        return [inv_vocab[i] for i in ints]

def one_step_attention_test():
    print(f"\n=== {one_step_attention_test.__name__} ===")
    m = 10000
    Tx = 30
    Ty = 10
    n_a = 32
    n_s = 64
    # size: 10000, Tx: 30, Ty: 10, n_a: 32, n_s: 64, X: (10000, 30), Y: (10000, 10), Xoh: (10000, 30, 37), Yoh: (10000, 10, 11)
    mt = MachineTranslation("en_SG", m, Tx, Ty, n_a, n_s)

    a = numpy.random.uniform(1, 0, (m, Tx, 2 * n_a)).astype(numpy.float32)
    s_prev = numpy.random.uniform(1, 0, (m, n_s)).astype(numpy.float32) * 1
    context = mt.one_step_attention(a, s_prev)
    
    expected_output = numpy.load('data/expected_output_ex1.npy')

    assert tf.is_tensor(context), "Unexpected type. It should be a Tensor"
    assert tuple(context.shape) == (m, 1, n_s), "Unexpected output shape"
    #assert numpy.all(context.numpy() == expected_output), "Unexpected values in the result"
    print("\033[92mAll tests passed!")

def model_test():
    print(f"\n=== {model_test.__name__} ===")
    m = 10000
    Tx = 30
    Ty = 10
    n_a = 32
    n_s = 64
    # size: 10000, Tx: 30, Ty: 10, n_a: 32, n_s: 64, X: (10000, 30), Y: (10000, 10), Xoh: (10000, 30, 37), Yoh: (10000, 10, 11)
    mt = MachineTranslation("en_SG", m, Tx, Ty, n_a, n_s)
    model = mt.BuildModel()

if __name__ == "__main__":
    print(tf.version.GIT_VERSION, tf.version.VERSION)
    one_step_attention_test()
    model_test()