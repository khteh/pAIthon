import numpy, tensorflow as tf
import IPython
import sys
import matplotlib.pyplot as plt
from music21 import *
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from outputs import *
from test_utils import *
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# https://github.com/cuthbertLab/music21/issues/1813
class LSTM_MusicGenerator():
    _model = None
    def __init__():
        pass
    def BuildModel(self):
        """
        Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
        
        Arguments:
        LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
        densor -- the trained "densor" from model(), Keras layer object
        Ty -- integer, number of time steps to generate
        
        Returns:
        inference_model -- Keras model instance
        """
        
        # Get the shape of input values
        n_values = densor.units
        # Get the number of the hidden state vector
        n_a = LSTM_cell.units
        
        # Define the input of your model with a shape 
        x0 = Input(shape=(1, n_values))
        
        
        # Define s0, initial hidden state for the decoder LSTM
        a0 = Input(shape=(n_a,), name='a0')
        c0 = Input(shape=(n_a,), name='c0')
        a = a0
        c = c0
        x = x0

        # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
        outputs = []
        
        # Step 2: Loop over Ty and generate a value at every time step
        for t in range(Ty):
            # Step 2.A: Perform one step of LSTM_cell. Use "x", not "x0" (≈1 line)
            _, a, c = LSTM_cell(inputs=x,initial_state=[a,c])
            
            # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
            out = densor(a)
            # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 90) (≈1 line)
            outputs.append(out)
    
            # Step 2.D: 
            # Select the next value according to "out",
            # Set "x" to be the one-hot representation of the selected value
            # See instructions above.
            #print(f"out: {out.shape}")
            x = tf.math.argmax(out, axis=-1)
            #print(f"argmax: {x}")
            x = tf.one_hot(x, n_values)
            # Step 2.E: 
            # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
            x = RepeatVector(1)(x)
            
        # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
        self._model = Model(inputs=[x0,a0,c0], outputs=outputs)