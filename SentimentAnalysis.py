import numpy,polars,timeit, pandas as pd
from pathlib import Path
from utils.FileUtil import Download, Unzip, Rename
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from utils.TensorModelPlot import PlotModelHistory
# Hide GPU from visible devices
def InitializeGPU():
    """
    2024-12-17 12:39:33.030218: I external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:1193] failed to allocate 2.2KiB (2304 bytes) from device: RESOURCE_EXHAUSTED: : CUDA_ERROR_OUT_OF_MEMORY: out of memory
    https://stackoverflow.com/questions/39465503/cuda-error-out-of-memory-in-tensorflow
    https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    https://www.tensorflow.org/guide/gpu
    """
    #tf.config.set_visible_devices([], 'GPU')
    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"{len(gpus)} GPUs available")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def BagOfWords():
    """
    Text sentence vectorization
    Each unique word will be assigned a unique index which is used to identify the words during training.
    Each sentence is converted into a list of numbers using the indices mapped in the vocabulary. This produces a feature vector for each sentence, a numerical representation of a sentence.
    Example:
    Vocabulary: {'John': 0, 'likes': 5, 'ice': 4, 'cream': 2, 'hates': 3, 'chocolate': 1}
    vector for 'John likes ice cream': [1,0,1,0,1,1] <- '1' indicates presence of the word; '0' indicates absence of the word in the array index which maps to the Vocabulary above
    vector for 'John hates chocolate': [1,1,0,1,0,0] <- '1' indicates presence of the word; '0' indicates absence of the word in the array index which maps to the Vocabulary above
    Problems with sparse matrix: If the vocabulary is huge, there is storage and memory waste since more of the sentences won't use most of the vocabulary and every sentence / vector produce has to have the length of the vocabulary
    Logistic regression can be used to simpkify / solve this sparse matrix issue.
    CountVectorizer creates a vector which has the size of the vocabulary
    """
    print(f"=== {BagOfWords.__name__} ===")
    sentences = ['John likes ice cream', 'John hates chocolate']
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer(min_df=1, lowercase=False) # Convert all to lowercase before tokenizing
    vectorizer.fit(sentences)
    print(f"Vocabulary: {vectorizer.vocabulary_}")
    data = vectorizer.transform(sentences).toarray() # This produces a sparse matrix
    print(f"\ndata ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    print(data)

def OneHotEncoding():
    """
    Word vectorization.
    The labels of the words maps to the index of the vector.
    Vocabulary: {"London": 1, "Berlin": 0, "New York": 2}
    Each vector has the same length as the voacbulary (unique wordss). A '1' maps to the label of the word.
    vector for "London": [0, 1, 0]
    vector for "Berlin": [1, 0, 0]
    vector for "New York": [0, 0, 1]
    Tokenizer creates a vector which has the size of the input text/sentence.
    """
    print(f"\n=== {OneHotEncoding.__name__} ===")
    cities = ['London', 'Berlin', 'Berlin', 'New York', 'London'] # 3 unique words.
    encoder = LabelEncoder()
    city_labels = encoder.fit_transform(cities)
    print(f"Word labels (LabelEncoder): {city_labels}")
    encoder = OneHotEncoder(sparse_output=False)
    city_labels = city_labels.reshape((-1,1)) # single column with every word taking a row
    print(f"Word labels: {city_labels}")
    city_labels = encoder.fit_transform(city_labels)
    print(f"Word labels (OneHotEncoder): {city_labels}") # Each vector has the same length as the voacbulary (unique wordss)

def CustomEmbeddingLayer(url, path):
    print(f"\n=== {CustomEmbeddingLayer.__name__} ===")
    Download(url, Path(path))
    Unzip(Path(path), "/tmp")
    Rename(Path("/tmp/sentiment labelled sentences"), "/tmp/sentiment_data")
    filepath_dict = {
        'yelp': '/tmp/sentiment_data/yelp_labelled.txt',
        'amazon': '/tmp/sentiment_data/amazon_cells_labelled.txt',
        'imdb': '/tmp/sentiment_data/imdb_labelled.txt'
    }
    dataframes = []
    columns = ["sentence", "label"]
    for source, path in filepath_dict.items():
        df = pd.read_csv(path, names=columns, sep="\t")
        df["source"] = source
        dataframes.append(df)
    # Combine data from all sources into a single DF
    data = pd.concat(dataframes).reset_index(drop=True)
    print(f"\ndata ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data.info()
    #print(data.head(20))
    sentences = data["sentence"]
    labels = data["label"]
    print(f"\nsentences ({id(sentences)}), ndim: {sentences.ndim}, size: {sentences.size}, shape: {sentences.shape}")
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=5678)
    tokenizer = Tokenizer(num_words=5000) # Assign an integer value to the 5000 most frequently used words
    tokenizer.fit_on_texts(sentences_train)
    # Create a vector for each vector in the datasets
    x_train = tokenizer.texts_to_sequences(sentences_train)
    x_test = tokenizer.texts_to_sequences(sentences_test)
    indices = sentences_train.index.tolist()
    for i in range(10):
        print(f"sentences_train[{indices[i]}]: {sentences_train[indices[i]]}")
        print(f"xtrain[{indices[i]}]         : {x_train[indices[i]]}")
    # Since the vectors of different lengths, need to be padded with zeroes to make axis-1 of the shape uniform
    x_train = pad_sequences(x_train, padding="post", maxlen=512)
    x_test = pad_sequences(x_test, padding="post", maxlen=512)
    # Add a custom embedding layer
    model = models.Sequential()
    model.add(
        layers.Embedding(
            input_dim=len(tokenizer.word_index) + 1, 
            output_dim=100, 
        )
    )
    """
    Low accuracy and high loss. Add a global pooling layer after the embedding layer in the model in order to
    consider the order of the values in the vectors. Inside the pooling layer, the max values in each dimension
    will be selected. There are also average pooling layers. The max pooling layer will highlight large values.
    """
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    #model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
                metrics=['accuracy'])
    history = model.fit(
        x_train, 
        y_train, 
        epochs=25,
        validation_data=(x_test, y_test)
    )
    print("Model Summary:") # This has to be done AFTER fit as there is no explicit Input layer added
    model.summary()
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=2)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f'Training accuracy: {train_accuracy:.4f}, loss: {train_loss:.4f}')
    print(f'Testing accuracy: {test_accuracy:.4f}, loss: {test_loss:.4f}')
    PlotModelHistory("CNN", history)
    """
    Training a real-world embedded layer takes a lot more time and attention. Better use pre-trained word embedding
    Some popular pretrained embeddings include Word2Vec from Google and GloVe from the NLP team at Standord Uni.
    """

def SentimentAnalysis(url, path):
    """
    Each unique word will be assigned a unique index which is used to identify the words during training.
    Each sentence is converted into a list of numbers using the indices mapped in the vocabulary. This produces a feature vector for each sentence, a numerical representation of a sentence.
    """
    print(f"\n=== {SentimentAnalysis.__name__} ===")
    Download(url, Path(path))
    Unzip(Path(path), "/tmp")
    Rename(Path("/tmp/sentiment labelled sentences"), "/tmp/sentiment_data")
    filepath_dict = {
        'yelp': '/tmp/sentiment_data/yelp_labelled.txt',
        'amazon': '/tmp/sentiment_data/amazon_cells_labelled.txt',
        'imdb': '/tmp/sentiment_data/imdb_labelled.txt'
    }
    dataframes = []
    columns = ["sentence", "label"]
    for source, path in filepath_dict.items():
        df = pd.read_csv(path, names=columns, sep="\t")
        df["source"] = source
        dataframes.append(df)
    # Combine data from all sources into a single DF
    data = pd.concat(dataframes).reset_index(drop=True)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data.info()
    print(data)
    sentences = data["sentence"]
    labels = data["label"]
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=5678)
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    x_train = vectorizer.transform(sentences_train).toarray()
    x_test = vectorizer.transform(sentences_test).toarray()
    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multinomial": "auto",
        "random_state": 8888,
    }
    classifier = LogisticRegression(**params)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    # Predict on the test set
    y_pred = classifier.predict(x_test)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"LogisticRegression: {len(sentences)} sentences, train shape: {x_train.shape}, score: {score}, accuracy: {accuracy}")

    model = models.Sequential()
    model.add(layers.Input(shape=(x_train.shape[1],)))  # Specify the input shape. https://keras.io/guides/sequential_model/#specifying-the-input-shape-in-advance
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print("Model Summary:")
    model.summary()
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
                metrics=['accuracy'])
    history = model.fit(
        x_train, 
        y_train, 
        epochs=25,
        validation_data=(x_test, y_test)
    )
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=2)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f'Training accuracy: {train_accuracy:.4f}, loss: {train_loss:.4f}')
    print(f'Testing accuracy: {test_accuracy:.4f}, loss: {test_loss:.4f}')
    PlotModelHistory("CountVectorizer", history)

if __name__ == "__main__":
    InitializeGPU()
    BagOfWords()
    OneHotEncoding()
    SentimentAnalysis("https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip", "/tmp/sentiment_data.zip")
    CustomEmbeddingLayer("https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip", "/tmp/sentiment_data.zip")
