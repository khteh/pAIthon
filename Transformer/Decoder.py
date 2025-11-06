import re,numpy, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, time, textwrap
from keras import saving
from tensorflow.keras.layers import Layer, Embedding, Dropout
from Transformer.positional_encoding import positional_encoding
from Transformer.masks import create_look_ahead_mask
from Transformer.DecoderLayer import DecoderLayer
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
wrapper = textwrap.TextWrapper(width=70)
@saving.register_keras_serializable()
class Decoder(Layer):
    """
    The entire Encoder starts by passing the target input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    decoder Layers
    """ 
    _embedding_dim : int = None
    _num_layers: int = None
    _embedding: Embedding = None
    _pos_encoding = None
    _dec_layers = None
    _dropout = None
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size, maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._embedding = Embedding(target_vocab_size, self._embedding_dim)
        self._pos_encoding = positional_encoding(maximum_position_encoding, self._embedding_dim)
        self._dec_layers = [DecoderLayer(embedding_dim=self._embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self._num_layers)]
        self._dropout = Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward  pass for the Decoder
        
        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len)
            enc_output (tf.Tensor):  Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input
            padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer
        Returns:
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attention_weights (dict[str: tf.Tensor]): Dictionary of tensors containing all the attention weights
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # create word embeddings 
        x = self._embedding(x)
        
        # scale embeddings by multiplying by the square root of their dimension
        x *= tf.math.sqrt(tf.cast(self._embedding_dim, tf.float32))
        
        # add positional encodings to word embedding
        x += self._pos_encoding[:,:seq_len,:]

        # apply a dropout layer to x
        # use `training=training`
        x = self._dropout(x, training=training)

        # use a for loop to pass x through a stack of decoder layers and update attention_weights (~4 lines total)
        for i in range(self._num_layers):
            # pass x and the encoder output through a stack of decoder layers and save the attention weights
            # of block 1 and 2 (~1 line)
            # def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
            x, block1, block2 = self._dec_layers[i](x, enc_output=enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

            #update attention_weights dictionary with the attention weights of block 1 and block 2
            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, fully_connected_dim)
        return x, attention_weights

def DecoderTests():
    # Test your function!
    n_layers = 5
    emb_d = 13
    n_heads = 17
    fully_connected_dim = 16
    target_vocab_size = 300
    maximum_position_encoding = 6

    x = numpy.array([[3, 2, 1, 1], [2, 1, 1, 0], [2, 1, 1, 0]])

    encoder_test_output = tf.convert_to_tensor(rng.uniform(size=(3, 7, 9)))

    look_ahead_mask = create_look_ahead_mask(x.shape[1])

    decoder_test = Decoder(n_layers, emb_d, n_heads, fully_connected_dim, target_vocab_size,maximum_position_encoding)
    # out, attn_w_b1, attn_w_b2 = decoderLayer_test(q, encoder_test_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=None)
    outd, att_weights = decoder_test(x, encoder_test_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=None)

    print(f"Using num_layers={n_layers}, embedding_dim={emb_d} and num_heads={n_heads}:\n")
    print(f"x has shape:{x.shape}")
    print(f"Output of encoder has shape:{encoder_test_output.shape}\n")

    print(f"Output of decoder has shape:{outd.shape}\n")
    print("Attention weights:")
    for name, tensor in att_weights.items():
        print(f"{name} has shape:{tensor.shape}")

if __name__ == "__main__":
    DecoderTests()