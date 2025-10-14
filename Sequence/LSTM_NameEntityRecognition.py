import os, numpy, pandas as pd, tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, LSTM, Embedding
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences

class LSTM_NameEntityRecognition():
    _path:str = None
    _train_sentences = None
    _train_labels = None
    _val_sentences = None
    _val_labels = None
    _test_sentences = None
    _test_labels = None
    _sentence_vectorizer = None
    _vocab = None
    _tags = None
    _tag_map = None
    def __init__(self, path):
        self._path = path
        self._PrepareData()
        
    def _PrepareData(self):
        self._train_sentences = self._load_data(f'{self._path}/large/train/sentences.txt')
        self._train_labels = self._load_data(f'{self._path}/large/train/labels.txt')

        self._val_sentences = self._load_data(f'{self._path}/large/val/sentences.txt')
        self._val_labels = self._load_data(f'{self._path}/large/val/labels.txt')

        self._test_sentences = self._load_data(f'{self._path}/large/test/sentences.txt')
        self._test_labels = self._load_data(f'{self._path}/large/test/labels.txt')
        self._sentence_vectorizer, self._vocab = self._get_sentence_vectorizer(self._train_sentences)
        self._tags = self._get_tags(self._train_labels)
        self._tag_map = self._make_tag_map(self._tags)

    def _get_sentence_vectorizer(self, sentences):
        tf.keras.utils.set_random_seed(33) ## Do not change this line. 
        """
        Create a TextVectorization layer for sentence tokenization and adapt it to the provided sentences.

        Parameters:
        sentences (list of str): Sentences for vocabulary adaptation.

        Returns:
        sentence_vectorizer (tf.keras.layers.TextVectorization): TextVectorization layer for sentence tokenization.
        vocab (list of str): Extracted vocabulary.
        """
        # Define TextVectorization object with the appropriate standardize parameter
        sentence_vectorizer = tf.keras.layers.TextVectorization(standardize=None)
        # Adapt the sentence vectorization object to the given sentences
        sentence_vectorizer.adapt(sentences)
        # Get the vocabulary
        vocab = sentence_vectorizer.get_vocabulary()      
        return sentence_vectorizer, vocab

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
        ### END CODE HERE ### 

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

if __name__ == "__main__":
    ner = LSTM_NameEntityRecognition("./data/NameEntityRecognition")