import numpy, argparse, pandas as pd, tensorflow as tf, json, random, logging, re
from pathlib import Path
from collections import Counter
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.optimizers import Adam
from tf_keras.optimizers import Adam
from pandas import DataFrame
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification #, TFDistilBertModel
from seqeval.metrics import classification_report
from utils.TensorModelPlot import PlotModelHistory
from utils.GPU import InitializeGPU, SetMemoryLimit
tf.get_logger().setLevel('ERROR')

class NameEntityRecognition():
    _path: str = None
    _model_path:str = None
    _data: DataFrame = None
    _cleaned_data = None
    _data_cleaned: DataFrame = None
    _unique_tags = None
    _id2tag = None
    _tag2id = None
    _maxlen:int = None
    _labels = None
    _tags = None
    _true_labels = None
    _tokenizer: DistilBertTokenizerFast = None
    _test = None
    _train = None
    _learning_rate:float = None
    _batch_size:int = None
    _epochs: int = None
    _model: TFDistilBertForTokenClassification = None
    def __init__(self, model_path:str, path:str, maxlen:int, learning_rate:float, batchsize:int, epochs:int):
        self._path = path
        self._model_path = model_path
        self._maxlen = maxlen
        self._learning_rate = learning_rate
        self._batch_size = batchsize
        self._epochs = epochs
        """
        Before feeding the texts to a Transformer model, you will need to tokenize your input using a 🤗 Transformer tokenizer.
        Tokenizer must match the Transformer model type!
        Use the 🤗 DistilBERT fast tokenizer, which standardizes the length of your sequence to 512 and pads with zeros.
        This matches the maximum length used when creating tags.
        """
        self._tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer/')
        self._PrepareData()
        #if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file(): TFDistilBertForTokenClassification is NOT decorated @keras.saving.register_keras_serializable()
        #    print(f"Using saved model {self._model_path}...")
        #    self._model = tf.keras.models.load_model(self._model_path)

    def BuildTrainModel(self, rebuild: bool = False):
        """
        Use DistilBERT model, which matches the tokenizer you used to preprocess your data.
        """
        print(f"\n=== {self.BuildTrainModel.__name__} ===")
        if self._model and not rebuild:
            return
        self._model = TFDistilBertForTokenClassification.from_pretrained('models/NameEntityRecognition', num_labels=len(self._unique_tags))
        # https://github.com/tensorflow/tensorflow/issues/100330
        # https://github.com/keras-team/keras/issues/21666
        self._model.compile(optimizer=Adam(learning_rate=self._learning_rate), loss=self._model.hf_compute_loss, metrics=['accuracy']) # can also use any keras loss fn
        history = self._model.fit(self._train.batch(self._batch_size), epochs=self._epochs, batch_size=self._batch_size)
        PlotModelHistory("Name Entity Recognition", history)
        #if self._model_path:
        #    self._model.save(self._model_path)
        #    print(f"Model saved to {self._model_path}.")

    def Predict(self, text:str):
        print(f"\n=== {self.Predict.__name__} ===")
        inputs = self._tokenizer(text, return_tensors="tf", truncation=True, is_split_into_words=False, padding="max_length", max_length=self._maxlen)
        predictions = self._model.predict(inputs)
        predictions = numpy.argmax(predictions['logits'].reshape(1, -1, 12), axis=-1)
        print(f"predictions: {predictions.shape}, {predictions}")
        pred_labels = [[self._id2tag.get(index, "Empty") for index in predictions[i]] for i in range(len(predictions))]
        p = plt.hist(numpy.array(pred_labels).flatten())
        plt.xticks(rotation='vertical')
        plt.show()

    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        self._data = pd.read_json(self._path, lines=True)
        self._data = self._data.drop(['extras'], axis=1)
        self._data['content'] = self._data['content'].str.replace("\n", " ")
        print("\ndata.info():")
        self._data.info()
        print("\nColumn labels:")
        print(self._data.columns)
        self._get_entities()
        self._trim_entity_spans(self._convert_dataturks_to_spacy(self._path))
        print(f"_cleaned_data: {len(self._cleaned_data)}")
        self._clean_dataset()
        self._generate_tags()
        self._labels = self._data_cleaned['setences_cleaned'].values.tolist()
        self._tags = pad_sequences([[self._tag2id.get(l) for l in lab] for lab in self._labels],
                            maxlen=self._maxlen, value=self._tag2id["Empty"], padding="post",
                            dtype="long", truncating="post")
        #print(f"{len(self._data['content'].values.tolist())} _data['content'].values")
        self._test = self._tokenize_and_align_labels(self._data['content'].values.tolist(), True)
        self._train = tf.data.Dataset.from_tensor_slices((
            self._test['input_ids'],
            self._test['labels']
        ))
        self._true_labels = [[self._id2tag.get(true_index, "Empty") for true_index in self._test['labels'][i]] for i in range(len(self._test['labels']))]
        #numpy.array(true_labels).shape
        p = plt.hist(numpy.array(self._true_labels).flatten())
        plt.xticks(rotation='vertical')
        plt.title("True Labels")
        plt.show()
        print(Counter(numpy.array(self._true_labels).flatten()))

    def _get_entities(self):
        print(f"\n=== {self._get_entities.__name__} ===")
        entities = []
        for i in range(len(self._data)):
            entity = []
            for annot in self._data['annotation'][i]:
                try:
                    ent = annot['label'][0]
                    start = annot['points'][0]['start']
                    end = annot['points'][0]['end'] + 1
                    entity.append((start, end, ent))
                except Exception as e:
                    #logging.exception(f"Error processing annotation! {e}")
                    pass
            entity = self._mergeIntervals(entity)
            entities.append(entity)
        self._data['entities'] = entities
   
    def _generate_tags(self):
        print(f"\n=== {self._generate_tags.__name__} ===")
        self._unique_tags = set(self._data_cleaned['setences_cleaned'].explode().unique())#pd.unique(self._data_cleaned['setences_cleaned'])#set(tag for doc in self._data_cleaned['setences_cleaned'].values.tolist() for tag in doc)
        self._tag2id = {tag: id for id, tag in enumerate(self._unique_tags)}
        self._id2tag = {id: tag for tag, id in self._tag2id.items()}        
        print(f"unique_tags: {self._unique_tags}")

    def _mergeIntervals(self, intervals):
        #print(f"\n=== {self._mergeIntervals.__name__} ===")
        sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
        merged = []

        for higher in sorted_by_lower_bound:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                if higher[0] <= lower[1]:
                    if lower[2] is higher[2]:
                        upper_bound = max(lower[1], higher[1])
                        merged[-1] = (lower[0], upper_bound, lower[2])
                    else:
                        if lower[1] > higher[1]:
                            merged[-1] = lower
                        else:
                            merged[-1] = (lower[0], higher[1], higher[2])
                else:
                    merged.append(higher)
        return merged
    
    def _convert_dataturks_to_spacy(self, dataturks_JSON_FilePath):
        print(f"\n=== {self._convert_dataturks_to_spacy.__name__} ===")
        try:
            training_data = []
            lines=[]
            with open(dataturks_JSON_FilePath, 'r') as f:
                lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                text = data['content'].replace("\n", " ")
                entities = []
                data_annotations = data['annotation']
                if data_annotations is not None:
                    for annotation in data_annotations:
                        #only a single point in text annotation.
                        point = annotation['points'][0]
                        labels = annotation['label']
                        # handle both list of labels or a single label.
                        if not isinstance(labels, list):
                            labels = [labels]

                        for label in labels:
                            point_start = point['start']
                            point_end = point['end']
                            point_text = point['text']
                            
                            lstrip_diff = len(point_text) - len(point_text.lstrip())
                            rstrip_diff = len(point_text) - len(point_text.rstrip())
                            if lstrip_diff != 0:
                                point_start = point_start + lstrip_diff
                            if rstrip_diff != 0:
                                point_end = point_end - rstrip_diff
                            entities.append((point_start, point_end + 1 , label))
                training_data.append((text, {"entities" : entities}))
            print(f"{len(training_data)} training data")
            return training_data
        except Exception as e:
            logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
            return None

    def _trim_entity_spans(self, data: list) -> list:
        """Removes leading and trailing white spaces from entity spans.

        Args:
            data (list): The data to be cleaned in spaCy JSON format.

        Returns:
            list: The cleaned data.
        """
        print(f"\n=== {self._trim_entity_spans.__name__} ===")
        invalid_span_tokens = re.compile(r'\s')
        self._cleaned_data = []
        for text, annotations in data:
            entities = annotations['entities']
            valid_entities = []
            for start, end, label in entities:
                valid_start = start
                valid_end = end
                while valid_start < len(text) and invalid_span_tokens.match(
                        text[valid_start]):
                    valid_start += 1
                while valid_end > 1 and invalid_span_tokens.match(
                        text[valid_end - 1]):
                    valid_end -= 1
                valid_entities.append([valid_start, valid_end, label])
            self._cleaned_data.append([text, {'entities': valid_entities}])

    def _clean_dataset(self):
        print(f"\n=== {self._clean_dataset.__name__} ===")
        self._data_cleaned = pd.DataFrame(columns=["setences_cleaned"])
        sum1 = 0
        print(f"Cleaning {len(self._cleaned_data)} data...")
        for i in tqdm(range(len(self._cleaned_data))):
            start = 0
            emptyList = ["Empty"] * len(self._cleaned_data[i][0].split())
            numberOfWords = 0
            lenOfString = len(self._cleaned_data[i][0])
            strData = self._cleaned_data[i][0]
            strDictData = self._cleaned_data[i][1]
            lastIndexOfSpace = strData.rfind(' ')
            for i in range(lenOfString):
                if (strData[i]==" " and strData[i+1]!=" "):
                    for k,v in strDictData.items():
                        for j in range(len(v)):
                            entList = v[len(v)-j-1]
                            if (start>=int(entList[0]) and i<=int(entList[1])):
                                emptyList[numberOfWords] = entList[2]
                                break
                            else:
                                continue
                    start = i + 1  
                    numberOfWords += 1
                if (i == lastIndexOfSpace):
                    for j in range(len(v)):
                            entList = v[len(v)-j-1]
                            if (lastIndexOfSpace>=int(entList[0]) and lenOfString<=int(entList[1])):
                                emptyList[numberOfWords] = entList[2]
                                numberOfWords += 1
            self._data_cleaned = pd.concat([self._data_cleaned, pd.Series([emptyList],  index=self._data_cleaned.columns).to_frame().T], ignore_index=True)
            sum1 = sum1 + numberOfWords

    def _tokenize_and_align_labels(self, data, label_all_tokens:bool = True):
        """
        Transformer models are often trained by tokenizers that split words into subwords. For instance, the word 'Africa' might get split into multiple subtokens. 
        This can create some misalignment between the list of tags for the dataset and the list of labels generated by the tokenizer, since the tokenizer can split one word into several, or add special tokens. 
        Before processing, it is important that you align the lists of tags and the list of labels generated by the selected tokenizer with a this function.

        The function performs the following:
        - The tokenizer cuts sequences that exceed the maximum size allowed by your model with the parameter truncation=True
        - Aligns the list of tags and labels with the tokenizer word_ids method returns a list that maps the subtokens to the original word in the sentence and special tokens to None.
        - Set the labels of all the special tokens (None) to -100 to prevent them from affecting the loss function.
        - Label of the first subtoken of a word and set the label for the following subtokens to -100.
        """
        print(f"\n=== {self._tokenize_and_align_labels.__name__} ===")
        tokenized_inputs = self._tokenizer(data, truncation=True, is_split_into_words=False, padding='max_length', max_length=self._maxlen)
        labels = []
        for i, label in enumerate(self._tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Content-based filtering recommendation system')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    InitializeGPU()
    SetMemoryLimit(4096)
    print(tf.version.GIT_VERSION, tf.version.VERSION)
    ner = NameEntityRecognition("models/name_entity_recognition.keras", "data/ner.json", 512, 1e-5, 4, 10)
    ner.BuildTrainModel(True) # TFDistilBertForTokenClassification is NOT decorated @keras.saving.register_keras_serializable()
    ner.Predict("Manisha Bharti. 3.5 years of professional IT experience in Banking and Finance domain")