import re,numpy, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, time, textwrap
from keras import saving
from tensorflow.keras.layers import Dropout, Embedding, Layer
from Transformer.EncoderLayer import EncoderLayer
from Transformer.positional_encoding import positional_encoding
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
wrapper = textwrap.TextWrapper(width=70)
@saving.register_keras_serializable()
class Encoder(Layer):
    """
    The entire Encoder starts by passing the input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    encoder Layers
    """
    _embedding_dim : int = None
    _num_layers: int = None
    _embedding: Embedding = None
    _pos_encoding = None
    _enc_layers = None
    _dropout = None
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._embedding = Embedding(input_vocab_size, self._embedding_dim)
        self._pos_encoding = positional_encoding(maximum_position_encoding, self._embedding_dim)
        self._enc_layers = [EncoderLayer(embedding_dim=self._embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self._num_layers)]
        self._dropout = Dropout(dropout_rate)
        
    def call(self, x, training, mask):
        """
        Forward pass for the Encoder
        
        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, seq_len)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            mask (tf.Tensor): Boolean mask to ensure that the padding is not 
                    treated as part of the input

        Returns:
            x (tf.Tensor): Tensor of shape (batch_size, seq_len, embedding dim)
        """
        #print(f"call() x shape: {tf.shape(x)}") [1 150]
        seq_len = tf.shape(x)[1]
        
        # Pass input through the Embedding layer
        x = self._embedding(x)  # (batch_size, input_seq_len, embedding_dim)
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x *= tf.math.sqrt(tf.cast(self._embedding_dim, tf.float32))
        # Add the position encoding to embedding
        x += self._pos_encoding[:, :seq_len, :]
        # Pass the encoded embedding through a dropout layer
        # use `training=training`
        x = self._dropout(x, training=training)
        # Pass the output through the stack of encoding layers 
        for i in range(self._num_layers):
            x = self._enc_layers[i](x, training=training, mask=mask)
        return x  # (batch_size, input_seq_len, embedding_dim)
