import argparse, json, string, numpy, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, time, textwrap, tensorflow_text as tf_text
from pathlib import Path
from tqdm import tqdm
from termcolor import colored
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
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
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
wrapper = textwrap.TextWrapper(width=70)

class QuestionAnswering():
    """
    Transformer Text Summarizer
    """
    _path:str = None
    _model_path:str = None
    _weights_path:str = None
    _encoder_maxlen: int = None
    _decoder_maxlen: int = None
    _batch_size:int = None
    _buffer_size:int = None
    # Data
    _dataset = None

    _train_json_dataset = None
    _train_data = None
    _train_targets = None
    _train_inputs: None
    _train_labels = None
    _train_labels = None

    _test_json_dataset = None
    _test_data = None
    _test_targets = None
    _test_inputs: None
    _test_labels = None
    _test_labels = None

    _filters: str = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n'
    _oov_token:str = '[UNK]'
    _vocab_size: int = None
    _learning_rate: float = None
    _tokenizer: Tokenizer = None
    # Define the model parameters
    _num_layers = 2
    _embedding_dim = 128
    _fully_connected_dim = 128
    _num_heads = 2
    _positional_encoding_length = 256

    _tokenizer = None
    _eos = None
    _tokenized_text = None
    _sentinels = None
    _inputs_targets_pairs = None
    _model: NLPTransformer = None
    _saved_model: bool = False
    _loss_object: SparseCategoricalCrossentropy = None
    _train_loss: Mean = None
    _optimizer = None
    _losses = None # Store the losses for plotting them against epochs
    def __init__(self, path, model_path:str, weights_path:str, num_layers:int, embedding_dim:int, fully_connected_dim:int, num_heads:int, positional_encoding_length:int, learning_rate: float, encoder_maxlen:int, decoder_maxlen:int, batch_size: int, buffer_size:int):
        self._path = path
        self._model_path = model_path
        self._weights_path = weights_path
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
        self._vocab_size = int(self._tokenizer.vocab_size())
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
            self._LoadWeights(self._weights_path) # Only load a pretrained weights on fresh model.
        self._model.summary()

    def TrainModel(self, epochs:int, retrain:bool = False):
        print(f"\n=== {self.TrainModel.__name__} ===")
        if not self._saved_model or retrain:
            self._LoadSquadDataset()
            #self._learning_rate = CustomSchedule(embedding_dim)
            self._optimizer = Adam(self._learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            self._loss_object = SparseCategoricalCrossentropy(from_logits=False, reduction='none') # The label is a single integer representing the index of the correct class.
            self._train_loss = Mean(name='train_loss')
            self._losses = [] # Here you will store the losses, so you can later plot them
            for epoch in tqdm(range(epochs)):
                start = time.time()
                self._train_loss.reset_states()
                number_of_batches=len(list(enumerate(self._dataset)))
                for (batch, (inp, tar)) in enumerate(self._dataset):
                    print(f'Epoch {epoch+1}, Batch {batch+1}/{number_of_batches}', end='\r')
                    self._model.train_step(inp, tar, self._loss_object, self._optimizer, self._train_loss)
                
                print (f'Epoch {epoch+1}, Loss {self._train_loss.result():.4f}')
                self._losses.append(self._train_loss.result())
                print (f'Time taken for one epoch: {time.time() - start} sec')
            plt.figure(figsize=(10, 10), constrained_layout=True)
            plt.plot(self._losses)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()
            plot_model(
                self._model,
                to_file="output/QuestionAnswering.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
            if self._model_path:
                self._model.save(self._model_path) #https://github.com/tensorflow/tensorflow/issues/102475
                print(f"Model saved to {self._model_path}.")

    def GetTestData(self, index):
        return self._test_inputs[index], self._test_labels[index]
    
    def answer_question(self, question):
        """
        A function for question answering using the transformer model
        Arguments:
            question (tf.Tensor): Input data with question and context
            model (tf.keras.model): The transformer model
            tokenizer (function): The SentencePiece tokenizer
            encoder_maxlen (number): Max length of the encoded sequence
            decoder_maxlen (number): Max length of the decoded sequence
        Returns:
            _ (str): The answer to the question
        """
        # QUESTION SETUP
        
        # Tokenize the question
        tokenized_question = self._tokenizer.tokenize(question)
        
        # Add an extra dimension to the tensor
        tokenized_question = tf.expand_dims(tokenized_question, 0) 
        
        # Pad the question tensor
        padded_question = pad_sequences(tokenized_question, maxlen=self._encoder_maxlen, padding='post', truncating='post') 
        # ANSWER SETUP
        
        # Tokenize the answer
        # Hint: All answers begin with the string "answer: "
        tokenized_answer = self._tokenizer.tokenize("answer: ")
        
        # Add an extra dimension to the tensor
        tokenized_answer = tf.expand_dims(tokenized_answer, 0)
        
        # Get the id of the EOS token
        eos = self._tokenizer.string_to_id("</s>") 
        
        # Loop for decoder_maxlen iterations
        for i in range(self._decoder_maxlen):
            
            # Predict the next word using the model, the input document and the current state of output
            next_word = self._model.NextWord(padded_question, tokenized_answer)
            
            # Concat the predicted next word to the output 
            tokenized_answer = tf.concat([tokenized_answer, next_word], axis=1)
            
            # The text generation stops if the model predicts the EOS token
            if next_word == eos:
                break 
        return tokenized_answer 
    
    def _LoadWeights(self, path:str):
        """
        Load a pretrained weights which was trained for a longer period of time. This saves time.
        """
        print(f"\n=== {self._LoadWeights.__name__} ===")
        if self._model and len(path) and Path(path).exists() and Path(path).is_file():
            self._model.load_weights(path)

    def _PrepareData(self):
        """
        The [C4 dataset](https://www.tensorflow.org/datasets/catalog/c4), also known as the Common Crawl C4 (Common Crawl Corpus C4), is a large-scale dataset of web pages collected by the [Common Crawl organization](https://commoncrawl.org/). 
        It is commonly used for various natural language processing tasks and machine learning research. Each sample in the C4 dataset follows a consistent format, making it suitable for pretraining models like BERT. Here's a short explanation and description of the C4 dataset:
        - Format: Each sample in the C4 dataset is represented as a JSON object, containing several key-value pairs.
        - Content: The 'text' field in each sample contains the actual text content extracted from web pages. This text often includes a wide range of topics and writing styles, making it diverse and suitable for training language models.
        - Metadata: The dataset includes metadata such as 'content-length,' 'content-type,' 'timestamp,' and 'url,' providing additional information about each web page. 'Content-length' specifies the length of the content, 'content-type' describes the type of content (e.g., 'text/plain'), 'timestamp' indicates when the web page was crawled, and 'url' provides the source URL of the web page.
        - Applications: The C4 dataset is commonly used for training and fine-tuning large-scale language models, such as BERT. It serves as a valuable resource for tasks like text classification, named entity recognition, question answering, and more.
        - Size: The C4 dataset is containing more than 800 GiB of text data, making it suitable for training models with billions of parameters.

        For the purpose of pretaining the T5 model, you will only use the `content` of each entry. In the following code, you filter only the field `text` from all the entries in the dataset. This is the data that you will use to create the `inputs` and `targets` of your language model.
        The [SentencePieceTokenizer](https://www.tensorflow.org/text/api_docs/python/text/SentencepieceTokenizer), used in the code snippet, tokenizes text into subword units, enhancing handling of complex word structures, out-of-vocabulary words, and multilingual support. 
        It simplifies preprocessing, ensures consistent tokenization, and seamlessly integrates with machine learning frameworks.
        In this task, a SentencePiece model is loaded from a file, which is used to tokenize text into subwords represented by integer IDs.        
        """
        print(f"\n=== {self._PrepareData.__name__} ===")
        # Load example jsons
        with open(f'{self._path}/c4-en-10k.jsonl', 'r') as file:
            self._json_dataset = [json.loads(line.strip()) for line in file]

        # Printing the examples to see how the data looks like
        for i in range(5):
            print(f'example number {i+1}: \n\n{self._json_dataset[i]} \n')
        
        # Grab text field from dictionary
        natural_language_texts = [example_json['text'] for example_json in self._json_dataset]

        # Print the first text example
        print(natural_language_texts[0])

        # Special tokens
        # PAD, EOS = 0, 1
        with open("./models/sentencepiece.model", "rb") as f:
            pre_trained_tokenizer = f.read()
            
        self._tokenizer = tf_text.SentencepieceTokenizer(pre_trained_tokenizer, out_type=tf.int32)
        self._eos = self._tokenizer.string_to_id("</s>").numpy()
        print("EOS: " + str(self._eos))
        self._tokenized_text = [(list(self._tokenizer.tokenize(word).numpy()), word) for word in natural_language_texts[2].split()]
        self._sentinels = self._get_sentinels(display=True)
        # Apply tokenize_and_mask
        self._inputs_targets_pairs = [self._tokenize_and_mask(text.encode('utf-8', errors='ignore').decode('utf-8')) 
                                for text in natural_language_texts[0:2000]]
        # Print 3 samples. We print inputs with less than 100 tokens. It is just to give you and idea of the process
        self._display_input_target_pairs(filter(lambda x: len(x[0]) < 100, self._inputs_targets_pairs[0:12]), wrapper)
        self._inputs = pad_sequences([x[0] for x in self._inputs_targets_pairs], maxlen=self._encoder_maxlen, padding='post', truncating='post')
        self._targets = pad_sequences([x[1] for x in self._inputs_targets_pairs], maxlen=self._decoder_maxlen, padding='post', truncating='post')

        self._inputs = tf.cast(self._inputs, dtype=tf.int32)
        self._targets = tf.cast(self._targets, dtype=tf.int32)

        # Create the final training dataset.
        self._dataset = tf.data.Dataset.from_tensor_slices((self._inputs, self._targets)).shuffle(self._buffer_size, reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
    def _LoadSquadDataset(self):
        """
        Now,  you are going to fine tune the pretrained model for Question Answering using the [SQUad 2.0 dataset](https://rajpurkar.github.io/SQuAD-explorer/).

        SQuAD, short for Stanford Question Answering Dataset, is a dataset designed for training and evaluating question answering systems. It consists of real questions posed by humans on a set of Wikipedia articles, where the answer to each question is a specific span of text within the corresponding article.

        SQuAD 1.1, the previous version of the SQuAD dataset, contains 100,000+ question-answer pairs on about 500 articles.
        SQuAD 2.0, contains 50.000 additional questions that are not meant to be answered. This extra set of questions can help to train models to detect unanswerable questions.
        """
        print(f"\n=== {self._LoadSquadDataset.__name__} ===")
        with open(f'{self._path}/train-v2.0.json', 'r') as f:
            self._train_json_dataset = json.load(f)
        self._train_json_dataset = self._train_json_dataset['data']
        print('Number of articles (training): ' + str(len(self._train_json_dataset)))
        self._train_data, self._train_targets = self._ParseSquad(self._train_json_dataset)
        print("Number of question/answer pairs: " + str(len(self._train_data)))
        print('\nFirst Q/A pair:\n\ninputs: ' + colored(self._train_data[0], 'blue'))
        print('\ntargets (training): ' + colored(self._train_targets[0], 'green'))
        print('\nLast Q/A pair:\n\ninputs: ' + colored(self._train_data[-1], 'blue'))
        print('\ntargets: ' + colored(self._train_targets[-1], 'green'))

        with open(f'{self._path}/dev-v2.0.json', 'r') as f:
            self._test_json_dataset = json.load(f)
        self._test_json_dataset = self._test_json_dataset['data']
        print('Number of articles (test): ' + str(len(self._test_json_dataset)))
        self._test_data, self._test_targets = self._ParseSquad(self._test_json_dataset)
        print("Number of question/answer pairs: " + str(len(self._test_data)))
        print('\nFirst Q/A pair:\n\ninputs: ' + colored(self._test_data[0], 'blue'))
        print('\ntargets (test): ' + colored(self._test_targets[0], 'green'))
        print('\nLast Q/A pair:\n\ninputs: ' + colored(self._test_data[-1], 'blue'))
        print('\ntargets: ' + colored(self._test_targets[-1], 'green'))

        # pairs for training
        inputs_str = [self._tokenizer.tokenize(s) for s in self._train_data]
        targets_str = [tf.concat([self._tokenizer.tokenize(s), [1]], 0) for s in self._train_targets]

        self._train_inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs_str, maxlen=self._encoder_maxlen, padding='post', truncating='post')
        self._train_labels = tf.keras.preprocessing.sequence.pad_sequences(targets_str, maxlen=self._decoder_maxlen, padding='post', truncating='post')

        self._train_inputs = tf.cast(self._train_inputs, dtype=tf.int32)
        self._train_labels = tf.cast(self._train_labels, dtype=tf.int32)
        
        self._dataset = tf.data.Dataset.from_tensor_slices((self._train_inputs, self._train_labels)).shuffle(self._buffer_size, reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        # pairs for training
        inputs_str = [self._tokenizer.tokenize(s) for s in self._test_data]
        targets_str = [tf.concat([self._tokenizer.tokenize(s), [1]], 0) for s in self._test_targets]

        self._test_inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs_str, maxlen=self._encoder_maxlen, padding='post', truncating='post')
        self._test_labels = tf.keras.preprocessing.sequence.pad_sequences(targets_str, maxlen=self._decoder_maxlen, padding='post', truncating='post')

        self._test_inputs = tf.cast(self._test_inputs, dtype=tf.int32)
        self._test_labels = tf.cast(self._test_labels, dtype=tf.int32)

    def _ParseSquad(self, dataset):
        """
        Generate input/output pairs for a Question Answering (QA) model using the SQuAD 2.0 dataset. Each pair follows the structure:

        - inputs: `question: <Q> context: <P>`
        - targets: `answer: <A>`
            
        Here, `<Q>` represents the question in the context of the given paragraph `<P>`, and `<A>` is a possible answer.

        In this notebook, we will focus on a single answer per question. However, it's essential to note that the dataset contains questions with multiple answers. When training a model in real-life scenarios, consider including all available information.

        <a name='ex-2'></a>
        ### Exercise 2 - Parse the SQuaD 2.0 Dataset

        Your task is to implement the parse_squad function, which iterates over all the articles, paragraphs, and questions in the SQuAD dataset. Extract pairs of inputs and targets for the QA model using the provided code template.
        - Start with two empty lists: `inputs` and `targets`.
        - Loop over all the articles in the dataset.
        - For each article, loop over each paragraph.
        - Extract the context from the paragraph.
        - Loop over each question in the given paragraph.
        - Check if the question is not impossible and has at least one answer.
        - If the above condition is met, create the `question_context` sequence as described in the input structure.
        - Create the `answer` sequence using the first answer from the available answers.
        - Append the `question_context` to the `inputs` list.
        - Append the `answer` to the `targets` list.        
        Extract all the answers/questions pairs from the SQuAD dataset

        Args:
            dataset (dict): The imported JSON dataset

        Returns:
            inputs, targets: Two lists containing the inputs and the targets for the QA model
        """
        print(f"\n=== {self._ParseSquad.__name__} ===")
        inputs, targets = [], []
        print(f"dataset: {type(dataset)} {len(dataset)}")
        # Loop over all the articles
        for article in dataset:
            #print(f"article: {article}")
            # Loop over each paragraph of each article
            for paragraph in article['paragraphs']:
                #print(f"paragraph: {paragraph}")
                # Extract context from the paragraph
                context = paragraph['context']
                
                #Loop over each question of the given paragraph
                for qa in paragraph['qas']:
                    
                    # If this question is not impossible and there is at least one answer
                    if len(qa['answers']) > 0 and not(qa['is_impossible']):
                        
                        # Create the question/context sequence
                        question_context = 'question: ' + qa['question'] + ' context: ' + paragraph['context']
                        
                        # Create the answer sequence. Use the text field of the first answer
                        answer = 'answer: ' + qa['answers'][0]['text']
                        
                        # Add the question_context to the inputs list
                        inputs.append(question_context)
                        
                        # Add the answer to the targets list
                        targets.append(answer)
        return inputs, targets

    def _display_input_target_pairs(self, inputs_targets_pairs, wrapper=textwrap.TextWrapper(width=70)):
        for i, inp_tgt_pair in enumerate(inputs_targets_pairs, 1):
            inps, tgts = inp_tgt_pair
            inps = str(self._pretty_decode(inps, self._sentinels, self._tokenizer).numpy(), encoding='utf-8')
            tgts = str(self._pretty_decode(tgts, self._sentinels, self._tokenizer).numpy(), encoding='utf-8')
            print(f'[{i}]\n\n'
                f'inputs:\n{wrapper.fill(text=inps)}\n\n'
                f'targets:\n{wrapper.fill(text=tgts)}\n\n\n')

    def _tokenize_and_mask(self, text, noise=0.15):
        """
        Tokenizes and masks input words based on a given probability. The probability is controlled by the `noise` parameter, typically set to mask around `15%` of the words in the input text. The function will generate two lists of tokenized sequences following the algorithm outlined below:
        - Start with two empty lists: `inps` and `targs`
        - Tokenize the input text using the given tokenizer.
        - For each `token` in the tokenized sequence:
        - Generate a random number(simulating a weighted coin toss)
        - If the random value is greater than the given threshold(noise):
            - Add the current token to the `inps` list
        - Else:
            - If a new sentinel must be included(read note **):
            - Compute the next sentinel ID using a progression.
            - Add a sentinel into the `inps` and `targs` to mark the position of the masked element.
            - Add the current token to the `targs` list.

        ** There's a special case to consider. If two or more consecutive tokens get masked during the process, you don't need to add a new sentinel to the sequences. To account for this, use the `prev_no_mask` flag, which starts as `True` but is turned to `False` each time you mask a new element. The code that adds sentinels will only be executed if, before masking the token, the flag was in the `True` state.

        Args:
            text (str or bytes): Text input.
            noise (float, optional): Probability of masking a token. Defaults to 0.15.
            randomizer (function, optional): Function that generates random values. Defaults to rng.uniform.
            tokenizer (function, optional): Tokenizer function. Defaults to tokenize.

        Returns:
            inps, targs: Lists of integers associated to inputs and targets.
        """
        # Current sentinel number (starts at 0)
        cur_sentinel_num = 0
        
        # Inputs and targets
        inps, targs = [], []

        # Vocab_size
        vocab_size = int(self._tokenizer.vocab_size())
        
        # EOS token id 
        # Must be at the end of each target!
        eos = self._tokenizer.string_to_id("</s>").numpy()
        
        # prev_no_mask is True if the previous token was NOT masked, False otherwise
        # set prev_no_mask to True
        prev_no_mask = True
        
        # Loop over the tokenized text
        for token in self._tokenizer.tokenize(text).numpy():
            # Generate a random value between 0 and 1
            rnd_val = rng.uniform() 
            
            # Check if the noise is greater than a random value (weighted coin flip)
            if noise > rnd_val:
                # Check if previous token was NOT masked
                if prev_no_mask:
                    # Current sentinel increases by 1
                    cur_sentinel_num += 1
                    # Compute end_id by subtracting current sentinel value out of the total vocabulary size
                    end_id = vocab_size - cur_sentinel_num
                    # Append end_id at the end of the targets
                    targs.append(end_id)
                    # Append end_id at the end of the inputs
                    inps.append(end_id)
                # Append token at the end of the targets
                targs.append(token)
                # set prev_no_mask accordingly
                prev_no_mask = False
            else:
                # Append token at the end of the inputs
                inps.append(token)
                # Set prev_no_mask accordingly
                prev_no_mask = True
        # Add EOS token to the end of the targets
        targs.append(eos)
        return inps, targs
    
    def _get_sentinels(self, display=False):
        """
        Create `input` and `target` pairs that will allow you to train your model. T5 uses the ids at the end of the vocab file as sentinels. For example, it will replace: 
        - `vocab_size - 1` by `<Z>`
        - `vocab_size - 2` by `<Y>`
        - and so forth. 
        
        It assigns every word a `chr`.
        """
        sentinels = {}
        vocab_size = self._tokenizer.vocab_size(name=None)
        for i, char in enumerate(reversed(string.ascii_letters), 1):
            decoded_text = self._tokenizer.detokenize([vocab_size - i]).numpy().decode("utf-8")
            
            # Sentinels, ex: <Z> - <a>
            sentinels[decoded_text] = f'<{char}>'    
        
            if display:
                print(f'The sentinel is <{char}> and the decoded token is:', decoded_text)
        return sentinels

    def _pretty_decode(self, encoded_str_list, sentinels):
        """
        Helps in handling the type when decoding.
        """
        # If already a string, just do the replacements.
        if tf.is_tensor(encoded_str_list) and encoded_str_list.dtype == tf.string:
            for token, char in sentinels.items():
                encoded_str_list = tf.strings.regex_replace(encoded_str_list, token, char)
            return encoded_str_list
    
        # We need to decode and then prettyfy it.
        return self._pretty_decode(self._tokenizer.detokenize(encoded_str_list), sentinels, self._tokenizer)
    
if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Transformer Question Answering')
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
    qa = QuestionAnswering("data/QuestionAnswering", "models/QuestionAnswering.keras", "models/QuestionAnswering/model_c4", 4, 128,128,4,256,0.0002, 150, 50, 64, 10000)
    qa.BuildModel()
    qa.TrainModel(10, args.retrain)

    idx = 1234
    question, label = qa.GetTestData(idx)
    result = qa.answer_question(idx)
    print(f"\nQuestion: {question}")
    print(f"Answer: {label}")
    print(f"Prediction: {result}")

    idx = 5678
    question, label = qa.GetTestData(idx)
    result = qa.answer_question(idx)
    print(f"\nQuestion: {question}")
    print(f"Answer: {label}")
    print(f"Prediction: {result}")

    idx = 90123
    question, label = qa.GetTestData(idx)
    result = qa.answer_question(idx)
    print(f"\nQuestion: {question}")
    print(f"Answer: {label}")
    print(f"Prediction: {result}")
