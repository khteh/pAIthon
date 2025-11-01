import argparse, os, re,numpy, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, time, textwrap
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean
from Transformer.EncoderLayer import EncoderLayer
from Transformer.Encoder import Encoder
from Transformer.Decoder import Decoder
from Transformer.DecoderLayer import DecoderLayer
from Transformer.NLPTransformer import NLPTransformer
from Transformer.CustomSchedule import CustomSchedule
from Transformer.masks import *
from utils.GPU import InitializeGPU
from utils.TermColour import bcolors
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
wrapper = textwrap.TextWrapper(width=70)

class TextSummarizer():
    """
    Transformer Text Summarizer
    """
    _path:str = None
    _model_path:str = None
    _encoder_maxlen: int = None
    _decoder_maxlen: int = None
    _batch_size:int = None
    _buffer_size:int = None
    _train_data = None
    _test_data = None
    _filters: str = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n'
    _oov_token:str = '[UNK]'
    _document = None
    _document_test = None
    _summary = None
    _summary_test = None
    _vocab_size: int = None
    _inputs: None
    _targets = None
    _dataset = None
    _learning_rate: float = None
    _tokenizer: Tokenizer = None
    # Define the model parameters
    _num_layers = 2
    _embedding_dim = 128
    _fully_connected_dim = 128
    _num_heads = 2
    _positional_encoding_length = 256

    _model: NLPTransformer = None
    _saved_model: bool = False
    _loss_object: SparseCategoricalCrossentropy = None
    _train_loss: Mean = None
    _optimizer = None
    _losses = None # Store the losses for plotting them against epochs
    def __init__(self, path, model_path:str, num_layers:int, embedding_dim:int, fully_connected_dim:int, num_heads:int, positional_encoding_length:int, learning_rate: float, encoder_maxlen:int, decoder_maxlen:int, batch_size: int, buffer_size:int):
        self._path = path
        self._model_path = model_path
        # Define the model parameters
        self._num_layers = num_layers
        self._embedding_dim = embedding_dim
        self._fully_connected_dim = fully_connected_dim
        self._num_heads = num_heads
        self._positional_encoding_length = positional_encoding_length
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        # Limit the size of the input and output data for being able to run it in this environment.
        self._encoder_maxlen = encoder_maxlen
        self._decoder_maxlen = decoder_maxlen
        self._PrepareData()
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._saved_model = True
            self._model = load_model(self._model_path) # https://github.com/tensorflow/tensorflow/issues/102475

    def BuildModel(self):
        print(f"\n=== {self.BuildModel.__name__} ===")
        if not self._model:
            # Initialize the model
            self._model = NLPTransformer(
                self._num_layers, 
                self._embedding_dim, 
                self._num_heads, 
                self._fully_connected_dim,
                self._vocab_size, 
                self._vocab_size, 
                self._positional_encoding_length, 
                self._positional_encoding_length,
            )
        self._model.summary()
    def TrainModel(self, epochs:int, retrain:bool = False):
        print(f"\n=== {self.TrainModel.__name__} ===")
        if not self._saved_model or retrain:
            #self._learning_rate = CustomSchedule(embedding_dim)
            self._optimizer = Adam(self._learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            self._loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            self._train_loss = Mean(name='train_loss')
            self._losses = [] # Here you will store the losses, so you can later plot them
            index = self._document.str.len().idxmax()
            document = self._document[index]
            true_summary = self._summary[index]
            for epoch in tqdm(range(epochs)):
                start = time.time()
                self._train_loss.reset_state()
                number_of_batches=len(list(enumerate(self._dataset)))
                for (batch, (inp, tar)) in enumerate(self._dataset):
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch+1}/{number_of_batches}', end='\r')
                    self._model.train_step(inp, tar, self._loss_object, self._optimizer, self._train_loss)
                self._losses.append(self._train_loss.result())
                # Take an example from the test set, to monitor it during training
                print (f'Epoch {epoch+1}/{epochs} {(time.time() - start):.2f}s Loss: {self._train_loss.result():.4f}')
                print(f"Document: {document}")
                print(f"{bcolors.OKGREEN}Expected Summarization:")
                print(f"  {true_summary}")
                print(f"{bcolors.WARNING}Predicted summarization:")
                print(f'  {self.Summarize(document)}{bcolors.DEFAULT}\n')
            fig = plt.figure(figsize=(20, 10)) # figsize = (width, height)
            plt.plot(self._losses)
            plt.ylabel('Loss', fontsize=22)
            plt.xlabel('Epoch', fontsize=22)
            plt.show()
            """
            plot_model(
                self._model,
                to_file="output/TransformerSummarizer.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
            """
            if self._model_path:
                self._model.save(self._model_path) #https://github.com/tensorflow/tensorflow/issues/102475
                print(f"Model saved to {self._model_path}.")

    def Predict(self, text:str):
        print(f"\n=== {self.Predict.__name__} ===")
        # Take a random sentence as an input
        input_document = self._tokenizer.texts_to_sequences([text])
        input_document = pad_sequences(input_document, maxlen=self._encoder_maxlen, padding='post', truncating='post')
        encoder_input = tf.expand_dims(input_document[0], 0)

        # Take the start of sentence token as the only token in the output to predict the next word
        output = tf.expand_dims([self._tokenizer.word_index["[SOS]"]], 0)

        # predict the next word with your function
        predicted_token = self._model.NextWord(encoder_input, output)
        print(f"Predicted token: {predicted_token}")
        return self._tokenizer.sequences_to_texts(predicted_token.numpy())[0]
    
    def GetTestDocument(self, index):
        return self._document_test[index], self._summary_test[index]

    def GetRandomTestDocument(self):
        choice = rng.choice(len(self._document_test), 1)[0]
        return self._document_test[choice], self._summary_test[choice]

    def Summarize(self, document:str):
        """
        A function for summarization using the transformer model
        Arguments:
            input_document (tf.Tensor): Input data to summarize
        Returns:
            _ (str): The summary of the input_document
        """    
        input_document = self._tokenizer.texts_to_sequences([document])
        input_document = pad_sequences(input_document, maxlen=self._encoder_maxlen, padding='post', truncating='post')
        encoder_input = tf.expand_dims(input_document[0], 0)
        output = tf.expand_dims([self._tokenizer.word_index["[SOS]"]], 0)
        for i in range(self._decoder_maxlen):
            predicted_id = self._model.NextWord(encoder_input, output)
            output = tf.concat([output, predicted_id], axis=-1)
            if predicted_id == self._tokenizer.word_index["[EOS]"]:
                break
        return self._tokenizer.sequences_to_texts(output.numpy())[0]  # since there is just one translated document
       
    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        # Get the train data
        self._train_data = pd.read_json(f"{self._path}/train.json")
        self._train_data.drop(['id'], axis=1, inplace=True)

        # Get the test data
        self._test_data = pd.read_json(f"{self._path}/test.json")
        self._test_data.drop(['id'], axis=1, inplace=True)

        # Take one example from the dataset and print it
        example_summary, example_dialogue = self._train_data.iloc[10]
        print(f"Dialogue:\n{example_dialogue}")
        print(f"\nSummary:\n{example_summary}")
        self._document, self._summary = self._Preprocess(self._train_data)
        self._document_test, self._summary_test = self._Preprocess(self._test_data)
        print(f"{len(self._document)} train documents ({type(self._document)}), {len(self._document_test)} test document")
        # The [ and ] from default tokens cannot be removed, because they mark the SOS and EOS token.
        self._tokenizer = Tokenizer(filters=self._filters, oov_token=self._oov_token, lower=False)
        documents_and_summary = pd.concat([self._document, self._summary], ignore_index=True)
        self._tokenizer.fit_on_texts(documents_and_summary)
        self._inputs = self._tokenizer.texts_to_sequences(self._document)
        self._targets = self._tokenizer.texts_to_sequences(self._summary)
        self._vocab_size = len(self._tokenizer.word_index) + 1
        print(f'Size of vocabulary: {self._vocab_size}')

        # Pad the sequences.
        self._inputs = pad_sequences(self._inputs, maxlen=self._encoder_maxlen, padding='post', truncating='post')
        self._targets = pad_sequences(self._targets, maxlen=self._decoder_maxlen, padding='post', truncating='post')

        self._inputs = tf.cast(self._inputs, dtype=tf.int32)
        self._targets = tf.cast(self._targets, dtype=tf.int32)

        # Create the final training dataset.
        self._dataset = tf.data.Dataset.from_tensor_slices((self._inputs, self._targets)).shuffle(self._buffer_size, reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def _Preprocess(self, input_data):
        # Define the custom preprocessing function
        def preprocess_util(input_data):
            # Convert all text to lowercase
            lowercase = input_data.lower()
            # Remove newlines and double spaces
            removed_newlines = re.sub("\n|\r|\t", " ",  lowercase)
            removed_double_spaces = ' '.join(removed_newlines.split(' '))
            # Add start of sentence and end of sentence tokens
            s = '[SOS] ' + removed_double_spaces + ' [EOS]'
            return s
        
        # Apply the preprocessing to the train and test datasets
        input_data['summary'] = input_data.apply(lambda row : preprocess_util(row['summary']), axis = 1)
        input_data['dialogue'] = input_data.apply(lambda row : preprocess_util(row['dialogue']), axis = 1)

        document = input_data['dialogue']
        summary = input_data['summary']
        return document, summary

    def _scaled_dot_product_attention(self, q, k, v, mask):
        """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.

        Arguments:
            q (tf.Tensor): query of shape (..., seq_len_q, depth)
            k (tf.Tensor): key of shape (..., seq_len_k, depth)
            v (tf.Tensor): value of shape (..., seq_len_v, depth_v)
            mask (tf.Tensor): mask with shape broadcastable 
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
            output -- attention_weights
        """
        # Multiply q and k transposed.
        matmul_qk = q @ k.T
        #print(f"k: {k.shape}")
        # scale matmul_qk with the square root of dk
        dk = tf.cast(k.shape[-1], tf.float32)
        #print(f"dk: {dk}")
        scaled_attention_logits = matmul_qk / numpy.sqrt(dk)
        #print(f"mask: {mask.shape}")
        # add the mask to the scaled tensor.
        if mask is not None:  # Don't replace this None
            scaled_attention_logits += tf.math.subtract(1.0, mask) * -1e9

        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        print(f"scaled_attention_logits: {scaled_attention_logits.shape}")
        attention_weights = tf.nn.softmax(scaled_attention_logits)

        # Multiply the attention weights by v
        output = attention_weights @ v
        return output, attention_weights    

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Transformer Text Summarizer')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    InitializeGPU()
    """
    _num_layers = 2
    _embedding_dim = 128
    _fully_connected_dim = 128
    _num_heads = 2
    _positional_encoding_length = 256
    """
    summarizer = TextSummarizer("data/corpus", "models/TextSummarizer.keras", 3, 128, 128, 3, 256, 0.0002, 150, 50, 64, 10000) # 4 num_layers and 5 num_heads will hit OOM error
    summarizer.BuildModel()
    summarizer.TrainModel(100, args.retrain)

    prediction = summarizer.Predict("A random sentence")
    print(f"Predicted word: {prediction}")
    document, human_summary = summarizer.GetRandomTestDocument()
    summary = summarizer.Summarize(document)
    print(f"Document: {document}")
    print(f"Human summary: {human_summary}")
    print(f"Model's summary: {summary}")