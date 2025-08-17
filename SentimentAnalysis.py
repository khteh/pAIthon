import numpy,polars,timeit, pandas as pd
from pathlib import Path
from utils.FileUtil import Download, Unzip, Rename
from utils.GPU import InitializeGPU
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, losses, optimizers, regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from utils.TensorModelPlot import PlotModelHistory

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
    model = models.Sequential([
        layers.Embedding(
            input_dim=len(tokenizer.word_index) + 1, 
            output_dim=100, 
        ),
        # Low accuracy and high loss. Add a global pooling layer after the embedding layer in the model in order to
        # consider the order of the values in the vectors. Inside the pooling layer, the max values in each dimension
        # will be selected. There are also average pooling layers. The max pooling layer will highlight large values.
        # L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
        # L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.      
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalMaxPool1D(),
        #layers.Flatten(),
        layers.Dense(10, activation='relu', name="L1", kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance.
        layers.Dense(1, activation='linear', name="L2") # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
    ]) 
    """
    In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
    These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
    Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
    It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.
    More stable and accurate results can be obtained if the sigmoid/softmax and loss are combined during training.
    In the preferred organization the final layer has a linear activation. For historical reasons, the outputs in this form are referred to as *logits*. The loss function has an additional argument: `from_logits = True`. This informs the loss function that the sigmoid/softmax operation should be included in the loss calculation. This allows for an optimized implementation.
    """
    model.compile(optimizer='adam', # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
                metrics=['accuracy'])
    # Epochs and batches
    # In the fit statement above, the number of epochs was set to 10. This specifies that the entire data set should be applied during training 10 times. During training, you see output describing the progress of training that looks like this:
    # Epoch 1/10
    # 6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
    # The first line, Epoch 1/10, describes which epoch the model is currently running. For efficiency, the training data set is broken into 'batches'. The default size of a batch in Tensorflow is 32. 
    # So, for example, if there are 200000 examples in our data set, there will be 6250 batches. The notation on the 2nd line 6250/6250 [==== is describing which batch has been executed.
    # Or, epochs = how many steps of a learning algorithm like gradient descent to run
    history = model.fit(
        x_train, 
        y_train, 
        epochs=25,
        validation_data=(x_test, y_test)
    )
    # Predict using linear activation with from_logits=True
    # This produces linear regression output (z). NOT g(z).
    logits = model.predict(x_test)
    f_x = tf.nn.sigmoid(logits) # g(z)
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
        #"multinomial": "auto",
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
    """
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.      
    """
    model = models.Sequential([
        layers.Input(shape=(x_train.shape[1],)),  # Specify the input shape. https://keras.io/guides/sequential_model/#specifying-the-input-shape-in-advance
        layers.Dense(10, activation='relu', name="L1", kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance.
        layers.Dense(1, activation='linear', name="L2")]) # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
    print("Model Summary:")
    model.summary()
    """
    In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
    These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
    Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
    It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.
    More stable and accurate results can be obtained if the sigmoid/softmax and loss are combined during training.
    In the preferred organization the final layer has a linear activation. For historical reasons, the outputs in this form are referred to as *logits*. The loss function has an additional argument: `from_logits = True`. This informs the loss function that the sigmoid/softmax operation should be included in the loss calculation. This allows for an optimized implementation.
    """
    model.compile(optimizer='adam', # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
                metrics=['accuracy'])
    # Epochs and batches
    # In the fit statement above, the number of epochs was set to 10. This specifies that the entire data set should be applied during training 10 times. During training, you see output describing the progress of training that looks like this:
    # Epoch 1/10
    # 6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
    # The first line, Epoch 1/10, describes which epoch the model is currently running. For efficiency, the training data set is broken into 'batches'. The default size of a batch in Tensorflow is 32. 
    # So, for example, if there are 200000 examples in our data set, there will be 6250 batches. The notation on the 2nd line 6250/6250 [==== is describing which batch has been executed.
    # Or, epochs = how many steps of a learning algorithm like gradient descent to run
    history = model.fit(
        x_train, 
        y_train, 
        epochs=25,
        validation_data=(x_test, y_test)
    )
    # Predict using linear activation with from_logits=True
    # This produces linear regression output (z). NOT g(z).
    logits = model.predict(x_test)
    f_x = tf.nn.sigmoid(logits)
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
