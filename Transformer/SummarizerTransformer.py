import re,numpy, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, time, textwrap
from tensorflow.keras import saving
from tensorflow.keras.utils import plot_model
from tensorflow.keras.saving import serialize_keras_object
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Layer, MultiHeadAttention, LayerNormalization
from tensorflow.keras.regularizers import l2
from Transformer.EncoderLayer import EncoderLayer
from Transformer.Encoder import Encoder
from Transformer.Decoder import Decoder
from Transformer.DecoderLayer import DecoderLayer
from Transformer.positional_encoding import positional_encoding
from Transformer.masks import create_look_ahead_mask, create_padding_mask

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
wrapper = textwrap.TextWrapper(width=70)
@saving.register_keras_serializable()
class SummarizerTransformer(Model):
    """
    Complete transformer with an Encoder and a Decoder
    """
    _num_layers: int = None
    _embedding_dim: int = None
    _num_heads: int = None
    _fully_connected_dim: int = None
    _input_vocab_size: int = None
    _target_vocab_size: int = None
    _max_positional_encoding_input: int = None
    _max_positional_encoding_target: int = None
    _dropout_rate: float = None
    _layernorm_eps: float = None
    _encoder: Encoder = None
    _decoder: Decoder = None
    _final_layer: Dense = None
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, target_vocab_size, max_positional_encoding_input, max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(SummarizerTransformer, self).__init__(**kwargs)
        self._num_layers = num_layers
        self._embedding_dim = embedding_dim
        self._num_heads = num_heads
        self._fully_connected_dim = fully_connected_dim
        self._input_vocab_size = input_vocab_size
        self._target_vocab_size = target_vocab_size
        self._max_positional_encoding_input = max_positional_encoding_input
        self._max_positional_encoding_target = max_positional_encoding_target
        self._dropout_rate = dropout_rate
        self._layernorm_eps = layernorm_eps
        self._encoder = Encoder(num_layers=self._num_layers,
                               embedding_dim=self._embedding_dim,
                               num_heads=self._num_heads,
                               fully_connected_dim=self._fully_connected_dim,
                               input_vocab_size=self._input_vocab_size,
                               maximum_position_encoding=self._max_positional_encoding_input,
                               dropout_rate=self._dropout_rate,
                               layernorm_eps=self._layernorm_eps)
        self._decoder = Decoder(num_layers=self._num_layers, 
                               embedding_dim=self._embedding_dim,
                               num_heads=self._num_heads,
                               fully_connected_dim=self._fully_connected_dim,
                               target_vocab_size=self._target_vocab_size, 
                               maximum_position_encoding=self._max_positional_encoding_target,
                               dropout_rate=self._dropout_rate,
                               layernorm_eps=self._layernorm_eps)
        self._final_layer = Dense(self._target_vocab_size, kernel_regularizer=l2(0.1)) # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected
    
    def call(self, input_sentence, output_sentence, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        Forward pass for the entire Transformer
        Arguments:
            input_sentence (tf.Tensor): Tensor of shape (batch_size, input_seq_len)
                              An array of the indexes of the words in the input sentence
            output_sentence (tf.Tensor): Tensor of shape (batch_size, target_seq_len)
                              An array of the indexes of the words in the output sentence
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            enc_padding_mask (tf.Tensor): Boolean mask to ensure that the padding is not 
                    treated as part of the input
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input
            dec_padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer
        Returns:
            final_output (tf.Tensor): The final output of the model
            attention_weights (dict[str: tf.Tensor]): Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        
        """
        # call self.encoder with the appropriate arguments to get the encoder output
        # def call(self, x, training, mask):
        enc_output = self._encoder(input_sentence, training=training, mask=enc_padding_mask)
        
        # call self.decoder with the appropriate arguments to get the decoder output
        # dec_output.shape == (batch_size, tar_seq_len, fully_connected_dim)
        # def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        dec_output, attention_weights = self._decoder(output_sentence, enc_output=enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
        
        # pass decoder output through a linear layer and softmax (~1 line)
        final_output = self._final_layer(dec_output)
        return final_output, attention_weights
    
    def get_config(self):
        """
        get_config(): This method should return a dictionary containing all the arguments needed to reconstruct an instance of your class.
        """
        base_config = super().get_config()
        config = {
            "num_layers": serialize_keras_object(self._num_layers),
            "embedding_dim": serialize_keras_object(self._embedding_dim),
            "num_heads": serialize_keras_object(self._num_heads),
            "fully_connected_dim": serialize_keras_object(self._fully_connected_dim),
            "input_vocab_size": serialize_keras_object(self._input_vocab_size),
            "target_vocab_size": serialize_keras_object(self._target_vocab_size),
            "max_positional_encoding_input": serialize_keras_object(self._max_positional_encoding_input),
            "max_positional_encoding_target": serialize_keras_object(self._max_positional_encoding_target),
            "dropout_rate": serialize_keras_object(self._dropout_rate),
            "layernorm_eps": serialize_keras_object(self._layernorm_eps),
            # Serialize any other custom objects or non-base type 
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        """
        from_config(): This is a class method that takes a configuration dictionary and returns a new instance of your class.
        """
        num_layers_config = config.pop("num_layers")
        num_layers = tf.keras.saving.deserialize_keras_object(num_layers_config)
        embedding_dim_config = config.pop("embedding_dim")
        embedding_dim = tf.keras.saving.deserialize_keras_object(embedding_dim_config)
        num_heads_config = config.pop("num_heads")
        num_heads = tf.keras.saving.deserialize_keras_object(num_heads_config)
        fully_connected_dim_config = config.pop("fully_connected_dim")
        fully_connected_dim = tf.keras.saving.deserialize_keras_object(fully_connected_dim_config)
        input_vocab_size_config = config.pop("input_vocab_size")
        input_vocab_size = tf.keras.saving.deserialize_keras_object(input_vocab_size_config)
        target_vocab_size_config = config.pop("target_vocab_size")
        target_vocab_size = tf.keras.saving.deserialize_keras_object(target_vocab_size_config)
        max_positional_encoding_input_config = config.pop("max_positional_encoding_input")
        max_positional_encoding_input = tf.keras.saving.deserialize_keras_object(max_positional_encoding_input_config)
        max_positional_encoding_target_config = config.pop("max_positional_encoding_target")
        max_positional_encoding_target = tf.keras.saving.deserialize_keras_object(max_positional_encoding_target_config)
        dropout_rate_config = config.pop("dropout_rate")
        dropout_rate = tf.keras.saving.deserialize_keras_object(dropout_rate_config)
        layernorm_eps_config = config.pop("layernorm_eps")
        layernorm_eps = tf.keras.saving.deserialize_keras_object(layernorm_eps_config)
        return cls(num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, target_vocab_size, max_positional_encoding_input, max_positional_encoding_target, dropout_rate, layernorm_eps, **config)
    
    def Plot(self):
        plot_model(
            self._encoder,
            to_file="output/TransformerSummarizerEncoder.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            show_layer_activations=True)
        plot_model(
            self._encoder,
            to_file="output/TransformerSummarizerDecoder.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            show_layer_activations=True)

def SummarizerTransformerTests():
    # Test your function!
    n_layers = 3
    emb_d = 13
    n_heads = 17
    fully_connected_dim = 8
    input_vocab_size = 300
    target_vocab_size = 350
    max_positional_encoding_input = 12
    max_positional_encoding_target = 12

    transformer = SummarizerTransformer(n_layers, 
        emb_d, 
        n_heads, 
        fully_connected_dim, 
        input_vocab_size, 
        target_vocab_size, 
        max_positional_encoding_input,
        max_positional_encoding_target)

    # 0 is the padding value
    sentence_a = numpy.array([[2, 3, 1, 3, 0, 0, 0]])
    sentence_b = numpy.array([[1, 3, 4, 0, 0, 0, 0]])

    enc_padding_mask = create_padding_mask(sentence_a)
    dec_padding_mask = create_padding_mask(sentence_a)

    look_ahead_mask = create_look_ahead_mask(sentence_a.shape[1])

    test_summary, att_weights = transformer(
        sentence_a,
        sentence_b,
        training=False,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=look_ahead_mask,
        dec_padding_mask=dec_padding_mask
    )

    print(f"Using num_layers={n_layers}, target_vocab_size={target_vocab_size} and num_heads={n_heads}:\n")
    print(f"sentence_a has shape:{sentence_a.shape}")
    print(f"sentence_b has shape:{sentence_b.shape}")

    print(f"\nOutput of transformer (summary) has shape:{test_summary.shape}\n")
    print("Attention weights:")
    for name, tensor in att_weights.items():
        print(f"{name} has shape:{tensor.shape}")

if __name__ == "__main__":
    SummarizerTransformerTests()