import re,numpy, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, time, textwrap
from tensorflow.keras import saving
from tensorflow.keras.utils import plot_model
from tensorflow.keras.saving import serialize_keras_object
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, Input, Layer, MultiHeadAttention, LayerNormalization
from tensorflow.keras.regularizers import l2
from Transformer.masks import create_look_ahead_mask
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

@saving.register_keras_serializable()
class GRULanguageModel(Model):
    """
    A GRU-based language model that maps from a tensor of tokens to activations over a vocabulary.

    - `tf.keras.layers.Embedding`: Initializes the embedding. In this case it is the size of the vocabulary by the dimension of the model. [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) 
    - `Embedding(vocab_size, embedding_dim)`.
    - `vocab_size` is the number of unique words in the given vocabulary.
    - `embedding_dim` is the number of elements in the word embedding (some choices for a word embedding size range from 150 to 300, for example).
    ___

    - `tf.keras.layers.GRU`: `TensorFlow` GRU layer. [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)) Builds a traditional GRU of rnn_units with dense internal transformations. You can read the paper here: https://arxiv.org/abs/1412.3555
        - `units`: Number of recurrent units in the layer. It must be set to `rnn_units`
        - `return_sequences`: It specifies if the model returns a sequence of predictions. Set it to `True`
        - `return_state`: It specifies if the model must return the last internal state along with the prediction. Set it to `True` 
    ___

    - `tf.keras.layers.Dense`: A dense layer. [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense). You must set the following parameters:
        - `units`: Number of units in the layer. It must be set to `vocab_size`
        - `activation`: It must be set to `log_softmax` function as described in the next line.
    ___

    - `tf.nn.log_softmax`: Log of the output probabilities. [docs](https://www.tensorflow.org/api_docs/python/tf/nn/log_softmax)
        - You don't need to set any parameters, just set the activation parameter as `activation=tf.nn.log_softmax`.
    ___

    Args:
        vocab_size (int, optional): Size of the vocabulary. Defaults to 256.
        embedding_dim (int, optional): Depth of embedding. Defaults to 256.
        rnn_units (int, optional): Number of units in the GRU cell. Defaults to 128.

    Returns:
        tf.keras.Model: A GRULM language model.
    """
    _vocab_size:int = None
    _embedding_dim:int = None
    _rnn_units: int = None
    def __init__(self, vocab_size, embedding_dim, rnn_units, **kwargs):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._rnn_units = rnn_units
        # Create an embedding layer to map token indices to embedding vectors
        self.embedding = Embedding(vocab_size, embedding_dim)
        # Define a GRU (Gated Recurrent Unit) layer for sequence modeling
        self.gru = GRU(rnn_units, return_sequences=True, return_state=True)
        # Apply a dense layer with log-softmax activation to predict next tokens
        #self.dense = Dense(vocab_size, activation=tf.nn.log_softmax, kernel_regularizer=l2(0.1)),  # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected)
        self.dense = Dense(vocab_size, kernel_regularizer=l2(0.1)),  # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected)
    
    def call(self, inputs, states=None, return_state=False, training=False):
        print(f"inputs: {inputs.shape}")
        x = inputs
        # Map input tokens to embedding vectors
        x = self.embedding(x, training=training)
        if states is None:
            # Get initial state from the GRU layer
            print(f"x: {x.shape}")
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        # Predict the next tokens and apply log-softmax activation
        x = self.dense(x, training=training)
        return x, states if return_state else x

    def get_config(self):
        """
        get_config(): This method should return a dictionary containing all the arguments needed to reconstruct an instance of your class.
        """
        base_config = super().get_config()
        config = {
            "embedding_dim": serialize_keras_object(self._embedding_dim),
            "vocab_size": serialize_keras_object(self._vocab_size),
            "rnn_units": serialize_keras_object(self._rnn_units),
            # Serialize any other custom objects or non-base type 
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        """
        from_config(): This is a class method that takes a configuration dictionary and returns a new instance of your class.
        """
        vocab_size_config = config.pop("vocab_size")
        vocab_size = tf.keras.saving.deserialize_keras_object(vocab_size_config)
        embedding_dim_config = config.pop("embedding_dim")
        embedding_dim = tf.keras.saving.deserialize_keras_object(embedding_dim_config)
        rnn_units_config = config.pop("rnn_units")
        rnn_units = tf.keras.saving.deserialize_keras_object(rnn_units_config)
        return cls(vocab_size, embedding_dim, embedding_dim, rnn_units**config)
