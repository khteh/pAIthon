import argparse, numpy, pandas as pd, tensorflow as tf
from pathlib import Path
from tensorflow.keras.utils import plot_model
from keras import saving
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, TextVectorization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from utils.TrainingMetricsPlot import PlotModelHistory
from utils.TrainingUtils import CreateTensorBoardCallback, CreateCircuitBreakerCallback

@saving.register_keras_serializable()
def masked_loss(y_true, y_pred):
    """
    Built-in accuracy metrics do not handle values which are to be ignored. For instance, the padded values.
    The model's prediction has 3 axes - (batch, #words, #classes)
    #words include padding to the longest sentence in the batch.
    #classes are the name entity tags.

    Calculate the masked sparse categorical cross-entropy loss.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.
    
    Returns:
    loss (tensor): Calculated loss.
    """
    # Calculate the loss for each item in the batch. Remember to pass the right arguments, as discussed above!
    # Since the last layer of the model finishes with a LogSoftMax call, the results are **not** normalized - they do not lie between 0 and 1. 
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, ignore_class=-1)
    # Use the previous defined function to compute the loss
    print(f"y_true: {y_true.shape}, pred: {y_pred.shape}")
    return loss_fn(y_true,y_pred)

@saving.register_keras_serializable()
def masked_accuracy(y_true, y_pred):
    """
    Calculate masked accuracy for predicted labels.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.

    Returns:
    accuracy (tensor): Masked accuracy.
    """
    # Calculate the loss for each item in the batch.
    # You must always cast the tensors to the same type in order to use them in training. Since you will make divisions, it is safe to use tf.float32 data type.
    y_true = tf.cast(y_true, tf.float32)
    # Create the mask, i.e., the values that will be ignored
    mask = tf.not_equal(y_true, -1)
    mask = tf.cast(mask, tf.float32) 
    #print(f"y_true: {y_true}, y_pred: {y_pred}, mask: {mask}")   
    
    # Perform argmax to get the predicted values
    y_pred_class = tf.math.argmax(y_pred, axis=-1)
    y_pred_class = tf.cast(y_pred_class, tf.float32)
    #print(f"y_pred_class: {y_pred_class}")
    # Compare the true values with the predicted ones
    matches_true_pred  = tf.equal(y_true, y_pred_class)
    matches_true_pred = tf.cast(matches_true_pred , tf.float32) 
    #print(f"matches_true_pred: {matches_true_pred}")
    # Multiply the acc tensor with the masks
    matches_true_pred *= mask
    #print(f"matches_true_pred unmasked: {matches_true_pred}")
    # Compute masked accuracy (quotient between the total matches and the total valid values, i.e., the amount of non-masked values)
    masked_acc = tf.math.divide(tf.reduce_sum(matches_true_pred), tf.reduce_sum(mask))
    #print(f"accuracy: {masked_acc}\n")
    return masked_acc

class LSTM_NameEntityRecognition():
    """
    LSTM Name Entity Recognition using dataset from Kaggle. It has been preprocessed. The original data consists of 4 columns: the sentence number, the word, the part-of-speech of the word (not used in this module), and the tags.
    A few tags expected to be used in this module:
    * geo: geographical entity
    * org: organization
    * per: person 
    * gpe: geopolitical entity
    * tim: time indicator
    * art: artifact
    * eve: event
    * nat: natural phenomenon
    * O: filler word

    The `tag_map` is a dictionary that maps the tags to numbers. The prepositions in the tags mean:
    * I: Token is inside an entity.
    * B: Token begins an entity.

    Example:

    **"Sharon flew to Miami on Friday"**

    The tags would look like:

    Sharon B-per
    flew   O
    to     O
    Miami  B-geo
    on     O
    Friday B-tim

    There are three tokens beginning with B-, since there are no multi-token entities in the sequence. But Sharon's last name is added to the sentence:

    **"Sharon Floyd flew to Miami on Friday"**

    Sharon B-per
    Floyd  I-per
    flew   O
    to     O
    Miami  B-geo
    on     O
    Friday B-tim

    Output tags would change to show first "Sharon" as B-per, and "Floyd" as I-per, where I- indicates an inner token in a multi-token sequence.
    """
    _path:str = None
    _model_path:str = None
    _embedding_dim: int = None
    _train_sentences = None
    _train_labels = None
    _train_label_vector = None
    _train_dataset = None

    _val_sentences = None
    _val_labels = None
    _val_label_vector = None
    _val_dataset = None

    _test_sentences = None
    _test_labels = None
    _test_label_vector = None
    _test_dataset = None

    _sentence_vectorizer = None
    _vocab = None
    _tags = None
    _tag_map = None
    model = None
    _circuit_breaker = None
    _batch_size:int = None
    _learning_rate: float = None
    def __init__(self, path, model_path:str, embedding_dimension: int = 50, learning_rate:float = 0.01, batch_size:int = 64):
        self._path = path
        self._model_path = model_path
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._embedding_dim = embedding_dimension
        self._PrepareData()
        self._circuit_breaker = CreateCircuitBreakerCallback("val_masked_accuracy", "max", 5)
        #if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
        #    print(f"Using saved model {self._model_path}...") 
        #    self.model = load_model(self._model_path) https://github.com/tensorflow/tensorflow/issues/102475
    
    def BuildTrainModel(self, epochs: int, retrain: bool = False):
        new_model = not self.model
        if not self.model:
            self.model = Sequential([
                Input(shape=(len(self._vocab),)),
                Embedding(len(self._vocab)+1, self._embedding_dim, mask_zero = True),
                LSTM(self._embedding_dim, return_sequences=True),
                Dense(len(self._tag_map), activation=tf.nn.log_softmax)
            ], name = 'NameEntityRecognition')
            self.model.compile(optimizer=tf.keras.optimizers.Adam(self._learning_rate), 
                        loss = masked_loss,
                        metrics = [masked_accuracy])
            self.model.summary()
            plot_model(
                self.model,
                to_file="output/LSTM_NameEntityRecognition.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
        if new_model or retrain:
            tensorboard = CreateTensorBoardCallback("LSTM_NameEntityRecognition") # Create a new folder with current timestamp
            # https://github.com/tensorflow/tensorflow/issues/103397
            history = self.model.fit(self._train_dataset.batch(self._batch_size),
                                        validation_data = self._val_dataset.batch(self._batch_size), shuffle=True, epochs = epochs, validation_freq=1, callbacks=[tensorboard, self._circuit_breaker]) # shuffle: Boolean, whether to shuffle the training data before each epoch. This argument is ignored when x is a generator or a tf.data.Dataset.
            PlotModelHistory("LSTM Name Entity Recognition", history)
            #if self._model_path:
            #    self.model.save(self._model_path) https://github.com/tensorflow/tensorflow/issues/102475
            #    print(f"Model saved to {self._model_path}.")

    def Evaluate(self):
        # Convert the sentences into ids
        test_sentences_id = self._sentence_vectorizer(self._test_sentences)
        # Rename to prettify next function call
        y_true = self._test_label_vector
        y_pred = self.model.predict(test_sentences_id)
        print(f"The model's accuracy in test set is: {masked_accuracy(y_true,y_pred).numpy():.4f}")

    def Predict(self, sentence:str):
        """
        Predict NER labels for a given sentence using a trained model.

        Parameters:
        sentence (str): Input sentence.
        model (tf.keras.Model): Trained NER model.
        sentence_vectorizer (tf.keras.layers.TextVectorization): Sentence vectorization layer.
        tag_map (dict): Dictionary mapping tag IDs to labels.

        Returns:
        predictions (list): Predicted NER labels for the sentence.
        """
        # Convert the sentence into ids
        sentence_vectorized = self._sentence_vectorizer(sentence)
        #print(f"sentence_vectorized: {sentence_vectorized.shape}")
        # Expand its dimension to make it appropriate to pass to the model
        sentence_vectorized = tf.expand_dims(sentence_vectorized, axis=0)
        #print(f"sentence_vectorized: {sentence_vectorized.shape}")
        # Get the model output
        output = self.model.predict(sentence_vectorized)
        #print(f"output: {output.shape}")
        # Get the predicted labels for each token, using argmax function and specifying the correct axis to perform the argmax
        outputs = numpy.argmax(output, axis = -1)
        #print(f"outputs: {outputs.shape} {outputs}")
        # Next line is just to adjust outputs dimension. Since this function expects only one input to get a prediction, outputs will be something like [[1,2,3]]
        # so to avoid heavy notation below, let's transform it into [1,2,3]
        outputs = outputs[0] 
        #print(f"outputs: {outputs.shape} {outputs}")
        # Get a list of all keys, remember that the tag_map was built in a way that each label id matches its index in a list
        labels = list(self._tag_map.keys())
        #print(f"labels: {labels}")
        pred = [] 
        # Iterating over every predicted token in outputs list
        for tag_idx in outputs:
            #print(f"tag_idx: {tag_idx}")
            pred_label = labels[tag_idx]
            #print(f"pred_label: {pred_label}")
            pred.append(pred_label)
        #print(f"pred: {pred}")
        return pred
       
    def _PrepareData(self):
        self._train_sentences = self._load_data(f'{self._path}/large/train/sentences.txt')
        self._train_labels = self._load_data(f'{self._path}/large/train/labels.txt')

        self._val_sentences = self._load_data(f'{self._path}/large/val/sentences.txt')
        self._val_labels = self._load_data(f'{self._path}/large/val/labels.txt')

        self._test_sentences = self._load_data(f'{self._path}/large/test/sentences.txt')
        self._test_labels = self._load_data(f'{self._path}/large/test/labels.txt')
        self._get_sentence_vectorizer(self._train_sentences)
        self._tags = self._get_tags(self._train_labels)
        self._tag_map = self._make_tag_map(self._tags)
        self._train_label_vector = self._label_vectorizer(self._train_labels, self._tag_map)
        self._val_label_vector = self._label_vectorizer(self._val_labels, self._tag_map)
        self._test_label_vector = self._label_vectorizer(self._test_labels, self._tag_map)
        self._train_dataset = self._generate_dataset(self._train_sentences, self._train_labels, self._sentence_vectorizer)
        self._val_dataset = self._generate_dataset(self._val_sentences, self._val_labels,  self._sentence_vectorizer)
        self._test_dataset = self._generate_dataset(self._test_sentences, self._test_labels,  self._sentence_vectorizer)

    def _generate_dataset(self, sentences, labels, sentence_vectorizer):
        sentences_ids = sentence_vectorizer(sentences)
        labels_ids = self._label_vectorizer(labels, tag_map = self._tag_map)
        dataset = tf.data.Dataset.from_tensor_slices((sentences_ids, labels_ids))
        return dataset
    
    def _get_sentence_vectorizer(self, sentences):
        """
        Create a TextVectorization layer for sentence tokenization and adapt it to the provided sentences.
        standardize parameter tells how the parser splits the sentences. By default, standardize = 'lower_and_strip_punctuation'. This may influence the NER task, since an upper case in the middle of a sentence may indicate an entity.
        Since the punctuations are also labelled, set standardize = None so everything will just be split into single tokens and mapped to a positive integer.

        Parameters:
        sentences (list of str): Sentences for vocabulary adaptation.

        Returns:
        sentence_vectorizer (tf.keras.layers.TextVectorization): TextVectorization layer for sentence tokenization.
        vocab (list of str): Extracted vocabulary.
        """
        # Define TextVectorization object with the appropriate standardize parameter
        self._sentence_vectorizer = TextVectorization(standardize=None)
        # Adapt the sentence vectorization object to the given sentences
        self._sentence_vectorizer.adapt(sentences)
        # Get the vocabulary
        self._vocab = self._sentence_vectorizer.get_vocabulary()

    def _label_vectorizer(self, labels, tag_map):
        """
        Convert list of label strings to padded label IDs using a tag mapping.

        Parameters:
        labels (list of str): List of label strings.
        tag_map (dict): Dictionary mapping tags to IDs.
        Returns:
        label_ids (numpy.ndarray): Padded array of label IDs.
        """
        label_ids = [] # It can't be a numpy array yet, since each sentence has a different size

        ### START CODE HERE ### 
        #print(f"labels: {labels}")
        # Each element in labels is a string of tags so for each of them:
        for element in labels:
            #print(f"element: {element}")
            # Split it into single tokens. You may use .split function for strings. Be aware to split it by a blank space!
            tokens = element.strip().split()
            #print(f"tokens: {tokens}")
            # Use the dictionaty tag_map passed as an argument to the label_vectorizer function
            # to make the correspondence between tags and numbers. 
            element_ids = []

            for token in tokens:
                element_ids.append(tag_map[token])

            # Append the found ids to corresponding to the current element to label_ids list
            label_ids.append(element_ids)
            
        # Pad the elements
        label_ids = tf.keras.utils.pad_sequences(label_ids, padding="post", value=-1)
        print(f"label_ids: {label_ids}")
        return label_ids
    
    def _get_tags(self, labels):
        tag_set = set() # Define an empty set
        for el in labels:
            for tag in el.split(" "):
                tag_set.add(tag)
        tag_list = list(tag_set) 
        tag_list.sort()
        return tag_list
    
    def _make_tag_map(self, tags):
        tag_map = {}
        for i,tag in enumerate(tags):
            tag_map[tag] = i 
        return tag_map    
    
    def _load_data(self, path:str):
        with open(path,'r') as file:
            data = numpy.array([line.strip() for line in file.readlines()])
        return data

def VerifyPaddingDoesNOTAffectAccuracy(ner: LSTM_NameEntityRecognition):
    """
    You will check now how padding does not affect the model's output. Of course the output dimension will change. If ten zeros are added at the end of the tensor, then the resulting output dimension will have 10 more elements (more specifically, 10 more arrays of length 17 each). 
    However, those are removed from any calculation further on, so it won't impact at all the model's performance and training. You will be using the function tf.expand_dims.
    """
    print(f"\n=== {VerifyPaddingDoesNOTAffectAccuracy.__name__} ===")
    x = tf.expand_dims(numpy.array([545, 467, 896]), axis = 0) # Expanding dims is needed to pass it to the model, 
                                                            # since it expects batches and not single prediction arrays
    x_padded = tf.expand_dims(numpy.array([545, 467, 896, 0, 0, 0]), axis = 0)
    pred_x = ner.model(x)
    pred_x_padded = ner.model(x_padded)
    print(f'x shape: {pred_x.shape}\nx_padded shape: {pred_x_padded.shape}')
    assert numpy.allclose(pred_x, pred_x_padded[:,:3,:])
    # Verify both return the same loss and accuracy
    y_true = tf.expand_dims([16, 6, 12], axis = 0)
    y_true_padded = tf.expand_dims([16,6,12,-1,-1,-1], axis = 0) # Remember you mapped the padded values to -1 in the labels
    assert numpy.allclose(masked_loss(y_true,pred_x), masked_loss(y_true_padded,pred_x_padded))
    assert numpy.allclose(masked_accuracy(y_true,pred_x), masked_accuracy(y_true_padded,pred_x_padded))
    print(f"masked_loss is the same: {numpy.allclose(masked_loss(y_true,pred_x), masked_loss(y_true_padded,pred_x_padded))}")
    print(f"masked_accuracy is the same: {numpy.allclose(masked_accuracy(y_true,pred_x), masked_accuracy(y_true_padded,pred_x_padded))}")    

def ModelValidationAndTest(ner: LSTM_NameEntityRecognition):
    print(f"\n=== {ModelValidationAndTest.__name__} ===")
    ner.Evaluate()
    sentence = "Peter Parker , the White House director of trade and manufacturing policy of U.S , said in an interview on Sunday morning that the White House was working to prepare for the possibility of a second wave of the coronavirus in the fall , though he said it wouldn ’t necessarily come"
    predictions = ner.Predict(sentence)
    for x,y in zip(sentence.split(' '), predictions):
        if y != 'O':
            print(f"{x}: {y}")

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='LSTM Name Entity Recognition')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    ner = LSTM_NameEntityRecognition("data/NameEntityRecognition", "models/lstm_name_entity_recognition.keras")
    ner.BuildTrainModel(100, args.retrain)
    VerifyPaddingDoesNOTAffectAccuracy(ner)
    ModelValidationAndTest(ner)