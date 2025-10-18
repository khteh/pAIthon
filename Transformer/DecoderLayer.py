import re,numpy, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, time, textwrap
from keras import saving
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Layer, MultiHeadAttention, LayerNormalization
from tensorflow.keras.regularizers import l2
from Transformer.masks import create_look_ahead_mask
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
wrapper = textwrap.TextWrapper(width=70)
@saving.register_keras_serializable()
class DecoderLayer(Layer):
    """
    The decoder layer is composed by two multi-head attention blocks, 
    one that takes the new input and uses self-attention, and the other 
    one that combines it with the output of the encoder, followed by a
    fully connected block. 
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )
        self.mha2 = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )
        self.ffn = Sequential([
                Dense(fully_connected_dim, activation='relu', kernel_regularizer=l2(0.1)),  # (batch_size, seq_len, d_model). Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected
                Dense(embedding_dim, kernel_regularizer=l2(0.1))  # (batch_size, seq_len, d_model)
            ])
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the Decoder Layer
        
        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            enc_output (tf.Tensor): Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input
            padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer
        Returns:
            out3 (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attn_weights_block1 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, target_seq_len)
            attn_weights_block2 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        # enc_output.shape == (batch_size, input_seq_len, fully_connected_dim)
        
        # BLOCK 1
        # calculate self-attention and return attention scores as attn_weights_block1.
        # Dropout will be applied during training (~1 line).
        mult_attn_out1, attn_weights_block1 = self.mha1(x,x,x, attention_mask=look_ahead_mask, return_attention_scores=True, training=training)
        
        # apply layer normalization (layernorm1) to the sum of the attention output and the input (~1 line)
        Q1 = self.layernorm1(mult_attn_out1 + x)

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output. 
        # Dropout will be applied during training
        # Return attention scores as attn_weights_block2 (~1 line)
        #print(f"enc_output: {enc_output.shape}")
        mult_attn_out2, attn_weights_block2 = self.mha2(query=Q1, key=enc_output,value=enc_output, return_attention_scores=True, training=training, attention_mask=padding_mask)
        
        # # apply layer normalization (layernorm2) to the sum of the attention output and the Q from the first block (~1 line)
        mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)
                
        #BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(mult_attn_out2)
        
        # apply a dropout layer to the ffn output
        # use `training=training`
        ffn_output = self.dropout_ffn(ffn_output)
        
        # apply layer normalization (layernorm3) to the sum of the ffn output and the output of the second block
        out3 = self.layernorm3(ffn_output + mult_attn_out2)
        return out3, attn_weights_block1, attn_weights_block2

def DecoderLayerTests():
    # Test your function!
    key_dim = 12
    n_heads = 16

    decoderLayer_test = DecoderLayer(embedding_dim=key_dim, num_heads=n_heads, fully_connected_dim=32)

    q = numpy.ones((1, 15, key_dim))
    encoder_test_output = tf.convert_to_tensor(numpy.random.rand(1, 7, 8))
    look_ahead_mask = create_look_ahead_mask(q.shape[1])

    out, attn_w_b1, attn_w_b2 = decoderLayer_test(q, encoder_test_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=None)

    print(f"Using embedding_dim={key_dim} and num_heads={n_heads}:\n")
    print(f"q has shape:{q.shape}")
    print(f"Output of encoder has shape:{encoder_test_output.shape}\n")

    print(f"Output of decoder layer has shape:{out.shape}")
    print(f"Att Weights Block 1 has shape:{attn_w_b1.shape}")
    print(f"Att Weights Block 2 has shape:{attn_w_b2.shape}")

if __name__ == "__main__":
    DecoderLayerTests()