from __future__ import print_function
import argparse, numpy, tensorflow as tf
from pathlib import Path
from utils.RNN_utils import *
from Softmax import softmax
from keras import saving
from tensorflow.keras.utils import plot_model
from tensorflow.strings import unicode_split, reduce_join
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Activation, Dropout, Input, GRU, LSTM, StringLookup
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.TrainingMetricsPlot import PlotModelHistory
from utils.shakespeare_utils import on_epoch_end, sample
from .GRULanguageModel import GRULanguageModel
from utils.GPU import InitializeGPU, SetMemoryLimit
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

@saving.register_keras_serializable()
def log_perplexity(y_true, y_pred):
    """
    Tells if a set of sentences are written by humans (lower score) instead of being randomly generated text from a machine.
    Function to calculate the log perplexity of a model.

    Args:
        preds (tf.Tensor): Predictions of a list of batches of tensors corresponding to lines of text.
        y_true (tf.Tensor): Actual list of batches of tensors corresponding to lines of text.

    Returns:
        float: The log perplexity of the model.
    """
    PADDING_ID: int = 1
    #print(f"preds: {y_pred.shape}, target: {y_true.shape}")
    ### START CODE HERE ###
    # Calculate log probabilities for predictions using one-hot encoding
    log_p = tf.math.reduce_sum(y_pred * tf.one_hot(y_true, y_pred.shape[-1]), axis= -1) # HINT: tf.one_hot(...) should replace one of the Nones
    #print(f"log_p: {log_p.shape}")
    # Identify non-padding elements in the target
    non_pad = 1.0 - tf.cast(tf.equal(tf.cast(PADDING_ID, tf.int64), tf.cast(y_true, tf.int64)), tf.float32)          # You should check if the target equals to PADDING_ID
    #print(f"log_p: {log_p.shape}, non_pad: {non_pad.shape}")
    # Apply non-padding mask to log probabilities to exclude padding
    log_p = log_p * non_pad                             # Get rid of the padding
    # Calculate the log perplexity by taking the sum of log probabilities and dividing by the sum of non-padding elements
    #print(f"log_p: {log_p.shape}, non_pad: {non_pad.shape}")
    log_ppx = tf.math.reduce_sum(log_p, axis=-1) / tf.math.reduce_sum(non_pad, axis=-1) # Remember to set the axis properly when summing up
    # Compute the mean of log perplexity
    log_ppx = tf.math.reduce_mean(log_ppx) # Compute the mean of the previous expression
    return -log_ppx

class GRU_CharacterGeneration():
    _path:str = None
    _model_path:str = None
    _data = None
    _vocab = None
    _train_data = None
    _val_data = None
    _ids = None
    _ids_dataset = None
    _eval_text = None
    _eval_ids = None
    _X_val = None
    _Y_val = None
    _batch_size: int = None
    _buffer_size: int = None
    _seq_length: int = None
    _train_dataset = None
    _val_dataset = None
    _embedding_dim: int = None
    _rnn_units: int = None
    _temperature: float = None
    _learning_rate: float = None
    _model: GRULanguageModel = None
    def __init__(self, path:str, model_path:str, rnn_units:int, embedding_dim: int, seq_length: int, batch_size: int, buffer_size:int, learning_rate:float, temperature: float):
        self._path = path
        self._model_path = model_path
        self._rnn_units = rnn_units
        self._embedding_dim = embedding_dim
        self._seq_length = seq_length
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._learning_rate = learning_rate
        self._temperature = temperature
        self._PrepareData()
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._saved_model = True
            self._model = tf.keras.models.load_model(self._model_path) # https://github.com/tensorflow/tensorflow/issues/102475
   
    def BuildTrainModel(self, epochs:int, retrain:bool = False):
        """
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
        """
        print(f"\n=== {self.BuildTrainModel.__name__} ===")
        new_model: bool = not self._model
        if not self._model:
            # https://discuss.ai.google.dev/t/problem-with-gru-stacking-in-text-generation-tutorial/28774/7
            input = Input(shape=(self._seq_length, ))
            initial_gru_state = Input((self._seq_length, self._embedding_dim))
            embedding = Embedding(len(self._vocab), self._embedding_dim)(input)
            sequences, states = GRU(self._rnn_units, return_sequences=True, return_state=True)(embedding, initial_state = initial_gru_state)
            #sequences, states = GRU(self._rnn_units, return_sequences=True, return_state=True)(embedding)
            output = Dense(len(self._vocab), activation=tf.nn.log_softmax, kernel_regularizer=l2(0.1))(sequences) # Using linear activation will hit "loss: nan - val_loss: nan"
            self._model = Model(input, [output, states])
            # def __init__(self, vocab_size=256, embedding_dim=256, rnn_units=128, **kwargs):
            # self._model = GRULanguageModel(len(self._vocab), self._embedding_dim, self._rnn_units)
            # Compile the model using the parametrized Adam optimizer and the SparseCategoricalCrossentropy funcion
            self._model.compile(optimizer=Adam(learning_rate=self._learning_rate), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=[log_perplexity])
        if new_model or retrain:
            history = self._model.fit(self._train_dataset, epochs=epochs, validation_data=self._val_dataset)
            PlotModelHistory("GRU Character Generation", history)
        self._model.summary()
        plot_model(
            self._model,
            to_file="output/GRU_CharacterGeneration.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            show_layer_activations=True)
        #if self._model_path:
        #    self._model.save(self._model_path) #https://github.com/tensorflow/tensorflow/issues/102475
        #    print(f"Model saved to {self._model_path}.")

    def Evaluate(self):
        print(f"\n=== {self.Evaluate.__name__} ===")
        #preds = self._model(tf.expand_dims(self._X_val, 0), training=False, states=None, return_state=True)
        preds = self._model.predict(self._val_dataset)
        print(f"predictions: {preds.shape}")
        #Get the log perplexity
        log_ppx = log_perplexity(tf.expand_dims(self._Y_val, 0), preds)
        print(f'The log perplexity and perplexity of your model are {log_ppx} and {numpy.exp(log_ppx)} respectively')

    def Generate(self, initial:str, max_len:int):
        print(f"\n=== {self.Generate.__name__} ===")
        #input_ids = self._line_to_tensor(initial)
        #print(f"initial: {len(initial)}, input_ids: {input_ids.shape}")
        #ids_dataset = tf.data.Dataset.from_tensor_slices(input_ids)
        #print(f"ids_dataset: {ids_dataset.shape}")
        return self._generate_n_chars(max_len, initial), '\n\n' + '_'*80

    @tf.function
    def _generate_one_step(self, inputs, states=None):
        """
        Generate a single character and update the model state.

        This function is your go-to method for generating a single character at a time. It accepts two key inputs: an initial input sequence and a state that can be thought of as the ongoing context or memory of the model. 
        The function delivers a single character prediction and an updated state, which can be used as the context for future predictions.

        Args:
            inputs (string): The input string to start with.
            states (tf.Tensor): The state tensor.

        Returns:
            tf.Tensor, states: The predicted character and the current GRU state.
        """
        # Convert strings to token IDs.
        
        # Transform the inputs into tensors
        input_ids = self._line_to_tensor(inputs)
        print(f"inputs: {inputs}, input_ids: {input_ids.shape} {input_ids}")
        # Predict the sequence for the given input_ids. Use the states and return_state=True
        # def call(self, inputs, states=None, return_state=False, training=False):
        #predicted_logits, states = self._model(input_ids, states, return_state=True)
        # predicted_logits, states = self._model(input_ids)
        #print(f"input_ids: {input_ids.shape}")
        #predicted_logits = self._model.predict(input_ids)
        #predicted_logits, status = self._model(tf.expand_dims(input_ids, 0), training=False, states=None, return_state=True)
        predicted_logits, states = self._model(input_ids, states=states, return_state=True)
        # Get only last element of the sequence
        predicted_logits = predicted_logits[0, -1, :]                      
        # Use the temperature_random_sampling to generate the next character. 
        # def temperature_random_sampling(log_probs, temperature=1.0):
        predicted_ids = self._temperature_random_sampling(predicted_logits)
        # Use the chars_from_ids to transform the code into the corresponding char
        predicted_chars = self._text_from_ids([predicted_ids], self.vocab)
        
        # Return the characters and model state.
        return tf.expand_dims(predicted_chars, 0), states
    
    def _generate_n_chars(self, num_chars, prefix):
        """
        Generate a text sequence of a specified length, starting with a given prefix.

        This function takes text generation to the next level. It orchestrates the iterative generation of a sequence of characters. At each iteration, generate_one_step is called with the last generated character and the most recent state. 
        This dynamic approach ensures that the generated text evolves organically, building upon the context and characters produced in previous steps. Each character generated in this process is collected and stored in the result list, forming the final output text.

        Args:
            num_chars (int): The length of the output sequence.
            prefix (string): The prefix of the sequence (also referred to as the seed).

        Returns:
            str: The generated text sequence.
        """
        states = None
        next_char = tf.constant([prefix])
        result = [next_char]
        for n in range(num_chars):
            next_char, states = self._generate_one_step(next_char, states=states)
            result.append(next_char)
        return tf.strings.join(result)[0].numpy().decode('utf-8')

    def _temperature_random_sampling(self, log_probs):
        """
        The GRULM model demonstrates an impressive ability to predict the most likely characters in a sequence, based on log scores. However, it's important to acknowledge that this model, in its default form, is deterministic and can result in repetitive and monotonous outputs. For instance, it tends to provide the same answer to a question consistently.
        To make your language model more dynamic and versatile, you can introduce an element of randomness into its predictions. This ensures that even if you feed the model in the same way each time, it will generate different sequences of text.
        To achieve this desired behavior, you can employ a technique known as random sampling. When presented with an array of log scores for the N characters in your dictionary, you add an array of random numbers to this data. The extent of randomness introduced into the predictions is regulated by a parameter called "temperature". 
        By comparing the random numbers to the original input scores, the model adapts its choices, offering diversity in its outputs.
        This doesn't imply that the model produces entirely random results on each iteration. Rather, with each prediction, there is a probability associated with choosing a character other than the one with the highest score.

        Temperature Random sampling from a categorical distribution. The higher the temperature, the more random the output. If temperature is close to 0, it means that the model will just return the index of the character with the highest input log_score
        
        Args:
            log_probs (tf.Tensor): The log scores for each characeter in the dictionary
            temperature (number): A value to weight the random noise. 
        Returns:
            int: The index of the selected character
        """
        # Generate uniform random numbers with a slight offset to avoid log(0)
        u = tf.random.uniform(minval=1e-6, maxval=1.0 - 1e-6, shape=log_probs.shape)
        
        # Apply the Gumbel distribution transformation for randomness
        g = -tf.math.log(-tf.math.log(u))
        
        # Adjust the logits with the temperature and choose the character with the highest score
        return tf.math.argmax(log_probs + g * self._temperature, axis=-1)
       
    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        self._data = [] # storing all the lines in a variable. 
        with open(self._path) as files:
            for line in files:        
                # remove leading and trailing whitespace
                pure_line = line.strip()#.lower()

                # if pure_line is not the empty string,
                if pure_line:
                    # append it to the list
                    self._data.append(pure_line)
        n_lines = len(self._data)
        print(f"Number of lines: {n_lines}")
        text = "\n".join(self._data)
        # The unique characters in the file
        self._vocab = sorted(set(text))
        self._vocab.insert(0,"[UNK]") # Add a special character for any unknown
        self._vocab.insert(1,"") # Add the empty character for padding.
        print(f'{len(self._vocab)} unique characters')
        print(" ".join(self._vocab))
        self._train_data = self._data[:-1000] # Leave the rest for training
        self._val_data = self._data[-1000:] # Create a holdout validation set
        print(f"Number of training lines: {len(self._train_data)}")
        print(f"Number of validation lines: {len(self._val_data)}")
        self._ids = self._line_to_tensor("\n".join(["Hello world!", "Generative AI"]))
        self._ids_dataset = tf.data.Dataset.from_tensor_slices(self._ids)
        self._train_dataset = self._create_batch_dataset(self._train_data)
        self._val_dataset = self._create_batch_dataset(self._val_data)
        self._eval_text = "\n".join(self._val_data)
        self._eval_ids = self._line_to_tensor([self._eval_text])
        self._X_val, self._Y_val = self._split_input_target(tf.squeeze(self._eval_ids, axis=0))

    def _create_batch_dataset(self, data):
        """
        Creates a batch dataset from a list of text lines.
        - Join all the input lines into a single string. When you have a big dataset, you would better use a flow from directory or any other kind of generator.
        - Transform your input text into numeric tensors
        - Create a TensorFlow DataSet from your numeric tensors: Just feed the numeric tensors into the function [`tf.data.Dataset.from_tensor_slices`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices)
        - Make the dataset produce batches of data that will form a single sample each time. This is, make the dataset produce a sequence of `seq_length + 1`, rather than single numbers at each time. You can do it using the `batch` function of the already created dataset. You must specify the length of the produced sequences (`seq_length + 1`). So, the sequence length produced by the dataset will `seq_length + 1`. It must have that extra element since you will get the input and the output sequences out of the same element. `drop_remainder=True` will drop the sequences that do not have the required length. This could happen each time that the dataset reaches the end of the input sequence.
        - Use the `split_input_target` to split each element produced by the dataset into the mentioned input and output sequences.The input will have the first `seq_length` elements, and the output will have the last `seq_length`. So, after this step, the dataset generator will produce batches of pairs (input, output) sequences.
        - Create the final dataset, using `dataset_xy` as the starting point. You will configure this dataset to shuffle the data during the generation of the data with the specified BUFFER_SIZE. For performance reasons, you would like that tensorflow pre-process the data in parallel with training. That is called [`prefetching`](https://www.tensorflow.org/guide/data_performance#prefetching), and it will be configured for you.

        Args:
            lines (list): A list of strings with the input data, one line per row.
            vocab (list): A list containing the vocabulary.
            seq_length (int): The desired length of each sample.
            batch_size (int): The batch size.

        Returns:
            tf.data.Dataset: A batch dataset generator.
        """
        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        
        # For simplicity, just join all lines into a single line
        single_line_data  = "\n".join(data)

        # Convert your data into a tensor using the given vocab
        all_ids = self._line_to_tensor(single_line_data)
        # Create a TensorFlow dataset from the data tensor
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        # Create a batch dataset
        data_generator = ids_dataset.batch(self._seq_length + 1, drop_remainder=True) 
        # Map each input sample using the split_input_target function
        dataset_xy = data_generator.map(self._split_input_target)
        
        # Assemble the final dataset with shuffling, batching, and prefetching
        dataset = (                                   
            dataset_xy                                
            .shuffle(self._buffer_size)
            .batch(self._batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)  
            )
        return dataset
    
    def _line_to_tensor(self, line):
        """
        Converts a line of text into a tensor of unicode integer values representing characters.

        Args:
            line (str): A single line of text.
            vocab (list): A list containing the vocabulary of unique characters.

        Returns:
            tf.Tensor(dtype=int64): A tensor containing integers (unicode values) corresponding to the characters in the `line`.
        """
        # Split the input line into individual characters
        chars = unicode_split(line, input_encoding='UTF-8')
        # Map characters to their respective integer values using StringLookup
        return StringLookup(vocabulary=list(self._vocab), mask_token=None)(chars)
    
    def _text_from_ids(self, ids):
        """
        Converts a tensor of integer values into human-readable text.

        Args:
            ids (tf.Tensor): A tensor containing integer values (unicode IDs).
            vocab (list): A list containing the vocabulary of unique characters.

        Returns:
            str: A string containing the characters in human-readable format.
        """
        # Initialize the StringLookup layer to map integer IDs back to characters
        chars_from_ids = StringLookup(vocabulary=self._vocab, invert=True, mask_token=None)
        
        # Use the layer to decode the tensor of IDs into human-readable text
        return reduce_join(chars_from_ids(ids), axis=-1)
    
    def _split_input_target(self, sequence):
        """
        Splits the input sequence into two sequences, where one is shifted by one position.

        Create 2 tensors, each with a length of `seq_length` out of the input sequence of lenght `seq_length + 1`. The first one contains the first `seq_length` elements and the second one contains the last `seq_length` elements. 
        For example, if you split the sequence `['H', 'e', 'l', 'l', 'o']`, you will obtain the sequences `['H', 'e', 'l', 'l']` and `['e', 'l', 'l', 'o']`.

        Args:
            sequence (tf.Tensor or list): A list of characters or a tensor.

        Returns:
            tf.Tensor, tf.Tensor: Two tensors representing the input and output sequences for the model.
        """
        # Create the input sequence by excluding the last character
        input_text = sequence[:-1]
        # Create the target sequence by excluding the first character
        target_text = sequence[1:]
        return input_text, target_text
    
    def test_GenerativeModel(self):
        print(f"\n=== {self.test_GenerativeModel.__name__} ===")
        self._temperature = 0.5
        n_chars = 40
        pre = "SEFOE"
        text1 = self._generate_n_chars(n_chars, pre)
        assert len(text1) == n_chars + len(pre) , f"Wrong length. Expected {n_chars + len(pre)} but got{len(text1)}"
        text2 = self._generate_n_chars(n_chars, pre)
        assert len(text2) == n_chars + len(pre), f"Wrong length. Expected {n_chars + len(pre)} but got{len(text2)}"
        assert text1 != text2, f"Expected different strings since temperature is > 0.0"

        self._temperature = 0.0
        n_chars = 40
        pre = "What is "
        text1 = self._generate_n_chars(n_chars, pre)
        assert len(text1) == n_chars + len(pre) , f"Wrong length. Expected {n_chars + len(pre)} but got{len(text1)}"
        text2 = self._generate_n_chars(n_chars, pre)
        assert len(text2) == n_chars + len(pre), f"Wrong length. Expected {n_chars + len(pre)} but got{len(text2)}"
        assert text1 == text2, f"Expected same strings since temperature is 0.0"
        
        n_chars = 100
        pre = "W"
        text_l = self._generate_n_chars(n_chars, pre)
        used_voc = set(text_l)
        assert used_voc.issubset(self._vocab), "Something went wrong. Only characters in vocab can be produced." \
        f" Unexpected characters: {used_voc.difference(self._vocab)}"
        print("\n\033[92mAll test passed!")
    
if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='GRU Character Generation')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    InitializeGPU()
    SetMemoryLimit(4096)
    #def __init__(self, path:str, model_path:str, rnn_units:int, embedding_dim: int, seq_length: int, batch_size: int, buffer_size:int, learning_rate:float, temperature: float):
    chargen = GRU_CharacterGeneration("data/shakespeare_data.txt", "models/GRU_CharacterGeneration.keras", 512, 256, 100, 64, 10000, 0.00125, 0.5)
    chargen.BuildTrainModel(10, args.retrain)
    #chargen.Evaluate() # OOM
    chargen.test_GenerativeModel()
    chargen.Generate("What's the meaning of life?", 1000)