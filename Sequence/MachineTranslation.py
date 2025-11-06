import argparse, numpy, tensorflow as tf, random, matplotlib.pyplot as plt, tensorflow.keras.backend as K
from datetime import datetime, date
from faker import Faker
from tqdm import tqdm
from babel.dates import format_date
from pathlib import Path
from keras import saving
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from utils.TrainingMetricsPlot import PlotModelHistory
from numpy.random import Generator, PCG64DXSM
from utils.TermColour import bcolors
rng = Generator(PCG64DXSM())

@saving.register_keras_serializable(name="_softmax")
def softmax(x, axis=1):
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

class MachineTranslation():
    """
    Nerual Machine Translation - Date translation
    Translate from human-readable date locale string to ISO-format date string.
    The model built in this module could be used to translate from one language into another. However, there is a caveat - language translation requires massive datasets and usually takes days of training on GPUs.
    """
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
    _path: str = None
    _fake: None
    _size:int = None # number of dates in the dataset
    _locale:str = None
    _Tx: int = None # length of the input sequence
    _Ty: int = None # length of the output sequence
    _dataset = None
    _human_vocab = None
    _machine_vocab = None
    _inv_machine_vocab = None
    _X = None
    _Y = None
    _Xoh = None
    _Yoh = None
    _n_a:int = None # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
    _n_s:int = None # number of units for the post-attention LSTM's hidden state "s"
    
    # The following are the optimizer's hyperparameters
    _learning_rate: float = None
    _beta_1: float = None
    _beta_2: float = None
    _batch_size: int = None
    _decay: float = None
    
    _repeator: RepeatVector = None
    _concatenator: Concatenate = None
    _densor1: Dense = None
    _densor2: Dense = None
    _activator: Activation = None
    _dot: Dot = None
    _model: Model = None
    _saved_model:bool = False
    _model_path:str = None
    _weights_path:str = None
    def __init__(self, path:str, weights_path: str, locale:str, size:int, tx:int, ty:int, n_a:int, n_s:int, learning_rate:float, beta1:float, beta2:float, decay:float, batchsize: int):
        self._locale = locale
        self._size = size
        self._Tx = tx
        self._Ty = ty
        self._n_a = n_a
        self._n_s = n_s
        self._learning_rate = learning_rate
        self._beta_1 = beta1
        self._beta_2 = beta2
        self._decay = decay
        self._batch_size = batchsize
        self._fake = Faker()
        self._PrepareData()
        # Defined shared layers as global variables
        self._repeator = RepeatVector(self._Tx)
        self._concatenator = Concatenate(axis=-1)
        self._densor1 = Dense(10, activation = "tanh", kernel_regularizer=l2(0.1)) # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected). tanh is similar to sigmoid graph but the output is [-1, 1]
        self._densor2 = Dense(1, activation = "relu", kernel_regularizer=l2(0.1))
        self._activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
        self._dot = Dot(axes = 1)
        self._model_path = path
        self._weights_path = weights_path
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._model = load_model(self._model_path)
            self._saved_model = True

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
        s_prev = self._repeator(s_prev)
        print(f"a.shape: {a.shape}, s_prev.shape: {s_prev.shape}")
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        # For grading purposes, please list 'a' first and 's_prev' second, in this order.
        concat = self._concatenator([a, s_prev])
        #print(f"a: {a.shape}, concat: {concat.shape}")
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = self._densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = self._densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = self._activator(energies) # We are using a custom softmax(axis = 1) loaded in this notebook. This layer produces the attention weights.
        # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = self._dot([alphas, a])
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
        if self._model:
            return
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

        # Please note, this is the post attention LSTM cell. These have to be REUSED in the following for loop instead of instantiating new layers.
        post_activation_LSTM_cell = LSTM(self._n_s, return_state = True) # Please do not modify this global variable.
        output_layer = Dense(len(self._machine_vocab), activation="softmax", kernel_regularizer=l2(0.1)) # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected

        # Step 2: Iterate for Ty steps
        for t in range(self._Ty):
        
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = self.one_step_attention(a, s)
            
            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector. (≈ 1 line)
            # Don't forget to pass: initial_state = [hidden state, cell state] 
            # Remember: s = hidden state, c = cell state
            # Remember to pass in the previous hidden-state  𝑠⟨𝑡−1⟩ and cell-states  𝑐⟨𝑡−1⟩ of this LSTM
            print(f"s: {s.shape}, c: {c.shape}, context: {context.shape}")
            _, s, c = post_activation_LSTM_cell(context, initial_state=[s, c])
            
            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = output_layer(s)
            
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
        self._model.compile(
                loss=CategoricalCrossentropy(from_logits=False), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
                optimizer=Adam(learning_rate=self._learning_rate, beta_1=self._beta_1, beta_2=self._beta_2, weight_decay=self._decay), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                metrics=['accuracy'] * self._Ty # https://github.com/tensorflow/tensorflow/issues/100319
            )
        self._LoadWeights(self._weights_path) # Only load a pretrained weights on fresh model.
        self._model.summary()
        plot_model(
            self._model,
            to_file="output/MachineTranslation.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            show_layer_activations=True)

    def Train(self, epochs:int, retrain: bool = False):
        print(f"\n=== {self.Train.__name__} ===")
        if not self._saved_model or retrain:
            s0 = numpy.zeros((self._size, self._n_s))
            c0 = numpy.zeros((self._size, self._n_s))
            outputs = list(self._Yoh.swapaxes(0,1))
            history = self._model.fit([self._Xoh, s0, c0], outputs, epochs=epochs, batch_size=self._batch_size)
            PlotModelHistory("Machine Translation", history)
            if self._model_path:
                self._model.save(self._model_path) # https://github.com/tensorflow/tensorflow/issues/100327
                print(f"Model saved to {self._model_path}.")

    def _LoadWeights(self, path:str):
        """
        Load a pretrained weights which was trained for a longer period of time. This saves time.
        """
        print(f"\n=== {self._LoadWeights.__name__} ===")
        if self._model and len(path) and Path(path).exists() and Path(path).is_file():
            self._model.load_weights(path)

    def Predict(self, dates, s00, c00):
        print(f"\n=== {self.Predict.__name__} ===")
        result = []
        for date in dates:
            source = self._string_to_int(date, self._Tx, self._human_vocab)
            #print(source)
            source = numpy.array(list(map(lambda x: to_categorical(x, num_classes=len(self._human_vocab)), source))).swapaxes(0,1)
            source = numpy.swapaxes(source, 0, 1)
            source = numpy.expand_dims(source, axis=0)
            prediction = self._model.predict([source, s00, c00])
            prediction = numpy.argmax(prediction, axis = -1) # (10, 1)
            output = [self._inv_machine_vocab[int(i.item())] for i in prediction]
            result.append(''.join(output))
        return result

    def visualize_attentions(self, text):
        """
        Plot the attention map.
        """
        print(f"\n=== {self.Predict.__name__} ===")
        attention_map = numpy.zeros((10, 30))
        Ty, Tx = attention_map.shape
        
        # Well, this is cumbersome but this version of tensorflow-keras has a bug that affects the 
        # reuse of layers in a model with the functional API. 
        # So, I have to recreate the model based on the functional 
        # components and connect then one by one.
        # ideally it can be done simply like this:
        # layer = self._model.layers[num]
        # f = Model(self._model.inputs, [layer.get_output_at(t) for t in range(Ty)])
        #
        X = self._model.inputs[0] 
        s0 = self._model.inputs[1] 
        c0 = self._model.inputs[2] 
        s = s0
        c = s0
        a = self._model.layers[2](X)  
        outputs = []
        for t in range(Ty):
            s_prev = s
            s_prev = self._model.layers[3](s_prev)
            concat = self._model.layers[4]([a, s_prev]) 
            e = self._model.layers[5](concat) 
            energies = self._model.layers[6](e) 
            alphas = self._model.layers[7](energies) 
            context = self._model.layers[8]([alphas, a])
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, _, c = self._model.layers[10](context, initial_state = [s, c]) 
            outputs.append(energies)

        f = Model(inputs=[X, s0, c0], outputs = outputs)
        s0 = numpy.zeros((1, self._n_s))
        c0 = numpy.zeros((1, self._n_s))
        encoded = numpy.array(self._string_to_int(text, Tx, self._human_vocab)).reshape((1, 30))
        encoded = numpy.array(list(map(lambda x: to_categorical(x, num_classes=len(self._human_vocab)), encoded)))
        r = f([encoded, s0, c0])
            
        for t in range(Ty):
            for t_prime in range(Tx):
                attention_map[t][t_prime] = r[t][0, t_prime]

        # Normalize attention map
        row_max = attention_map.max(axis=1)
        attention_map = attention_map / row_max[:, None]

        prediction = self._model.predict([encoded, s0, c0])
        #print(f"prediction: {type(prediction)}, {prediction[0]}")
        predicted_text = []
        for i in range(len(prediction)):
            predicted_text.append(int(numpy.argmax(prediction[i], axis=-1).item()))
            
        predicted_text = list(predicted_text)
        predicted_text = self._int_to_string(predicted_text, self._inv_machine_vocab)
        text_ = list(text)
        
        # get the lengths of the string
        input_length = len(text)
        output_length = Ty

        # Plot the attention_map
        fig, ax = plt.subplots(1,1, constrained_layout=True, figsize=(20, 10)) # figsize = (width, height)
        # Use tight_layout with h_pad to adjust vertical padding
        # Adjust h_pad for more/less vertical space
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) #[left, bottom, right, top]

        # add image
        img = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')
        # Create an axes divider from the main axes
        divider = make_axes_locatable(ax)

        # Append a new axes for the colorbar at the "bottom", with a specific width and padding
        # The 'size' parameter controls the colorbar's thickness, and 'pad' controls the distance
        cax = divider.append_axes("bottom", size="5%", pad=1)
        cbar = fig.colorbar(img, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=18) # Change 14 to your desired font size
        cbar.set_label("Alpha value (Probability output of the 'softmax')", fontsize=20) # Change 16 to your desired font size

        # add labels
        ax.set_yticks(range(output_length))
        ax.set_yticklabels(predicted_text[:output_length])

        ax.set_xticks(range(input_length))
        ax.set_xticklabels(text_[:input_length], rotation=45)

        ax.set_xlabel('Input Sequence', fontsize=20)
        ax.set_ylabel('Output Sequence', fontsize=20)

        # Adjust the left margin to create more space
        plt.subplots_adjust(left=0.05) # Increase 'left' value for more space

        plt.xticks(fontsize=18) 
        plt.yticks(fontsize=18) 
        # add grid and legend
        ax.grid()
        fig.suptitle("Attention Map", y=0.95, fontsize=22, fontweight="bold")
        plt.show()
        #return attention_map

    def ModelStateTest(self):
        """
        Check if the model correctly updates the (next) `hidden state` and `cell state`.
        """
        print(f"\n=== {self.ModelStateTest.__name__} ===")
        # Create test inputs
        X_test = rng.random((1, self._Tx, len(self._human_vocab)))
        s0_test = numpy.zeros((1, self._n_s))
        c0_test = numpy.zeros((1, self._n_s))

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, self._Tx, len(self._human_vocab)], dtype=tf.float32),
            tf.TensorSpec(shape=[None, self._n_s], dtype=tf.float32),
            tf.TensorSpec(shape=[None, self._n_s], dtype=tf.float32)
        ])
        def predict_function(X, s0, c0):
            # Call the model directly with input tensors
            return self._model([X, s0, c0])  

        # Get the outputs of the model for the first five time steps
        outputs = predict_function(X_test, s0_test, c0_test)

        # Extract the hidden states (s) from the LSTM outputs for each time step
        hidden_states = [numpy.array(output) for output in outputs]

        # Check if consecutive hidden states are different
        for i in range(len(hidden_states) - 1):
            assert not numpy.allclose(hidden_states[i], hidden_states[i + 1]), (
                "Consecutive hidden states should be different.\n"
                "Check if the LSTM cell is using the correct previous states.\n"
                "Make sure you are using s and c, and NOT using s0 and c0 in Step 2.B."
            )
    
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
            human_readable = format_date(dt, format=rng.choice(self.FORMATS),  locale=self._locale)
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
    # def __init__(self, path: str, locale:str, size:int, tx:int, ty:int, n_a:int, n_s:int, learning_rate:float, beta1:float, beta2:float, decay:float, batchsize: int):
    # lr=0.005, beta_1=0.9,beta_2=0.999,decay=0.01
    mt = MachineTranslation("models/MachineTranslation.keras", "models/machine_translation_weights.h5", "en_SG", m, Tx, Ty, n_a, n_s, 0.005, 0.9, 0.999,0.01, 100)

    a = rng.uniform(low=0, high=1, size=(m, Tx, 2 * n_a)).astype(numpy.float32)
    s_prev = rng.uniform(low=0, high=1, size=(m, n_s)).astype(numpy.float32) * 1
    context = mt.one_step_attention(a, s_prev)
    
    #expected_output = numpy.load('data/expected_output_ex1.npy')

    assert tf.is_tensor(context), "Unexpected type. It should be a Tensor"
    assert tuple(context.shape) == (m, 1, n_s), "Unexpected output shape"
    #assert numpy.all(context.numpy() == expected_output), "Unexpected values in the result"
    print(f"{bcolors.OKGREEN}All tests passed!{bcolors.DEFAULT}")

def model_test(retrain:bool):
    print(f"\n=== {model_test.__name__} ===")
    m = 10000
    Tx = 30
    Ty = 10
    n_a = 32
    n_s = 64
    # size: 10000, Tx: 30, Ty: 10, n_a: 32, n_s: 64, X: (10000, 30), Y: (10000, 10), Xoh: (10000, 30, 37), Yoh: (10000, 10, 11)
    # def __init__(self, path: str, locale:str, size:int, tx:int, ty:int, n_a:int, n_s:int, learning_rate:float, beta1:float, beta2:float, decay:float, batchsize: int):
    # lr=0.005, beta_1=0.9,beta_2=0.999,decay=0.01
    mt = MachineTranslation("models/MachineTranslation.keras", "models/machine_translation_weights.h5", "en_SG", m, Tx, Ty, n_a, n_s, 0.005, 0.9, 0.999,0.01, 100) # Increasing epochs does not improve accuracy. Have to examine the training dataset!
    mt.BuildModel()
    mt.ModelStateTest()
    mt.Train(100, retrain)
    print(f"date.today(): {date.today()}")
    print(f"datetime.now().date: {datetime.now().date()}")
    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2021', 'March 3rd 2001', '1 March 2001', '4th May 2023', "25th December 2025", "31st October 2021", "3rd November 2022"]
    expected = ["1979-05-03", "2009-04-05", "2016-08-21", "2007-07-10", "2018-05-09", "2021-03-03", "2001-03-03", "2001-03-01", "2023-05-04", "2025-12-25", "2021-10-31", "2022-11-03"]
    s00 = numpy.zeros((1, n_s))
    c00 = numpy.zeros((1, n_s))
    predictions = mt.Predict(EXAMPLES, s00, c00)
    for d, truth, prediction in zip(EXAMPLES, expected, predictions):
        print(f"\nsource: {d}")
        print(f"{bcolors.OKGREEN if truth == prediction else bcolors.FAIL}Prediction: {prediction}{bcolors.DEFAULT}")
    mt.visualize_attentions("25th December 2025")

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Neural Machine Translation - Date translation')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    print(tf.version.GIT_VERSION, tf.version.VERSION)
    one_step_attention_test()
    model_test(args.retrain) # https://github.com/tensorflow/tensorflow/issues/100327