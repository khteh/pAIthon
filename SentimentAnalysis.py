import numpy,polars,timeit, pandas as pd
from pathlib import Path
from utils.FileUtil import Download, Unzip, Rename
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from utils.TensorModelPlot import plot_history
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
    data = pd.concat(dataframes)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data.info()
    print(data)
    for source in data['source'].unique():
        print(f"\nsource: {source}")
        data_source = data[data['source'] == source]
        sentences = data_source["sentence"]
        labels = data_source["label"]
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=5678)
        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_train)
        x_train = vectorizer.transform(sentences_train).toarray()
        x_test = vectorizer.transform(sentences_test).toarray()
        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
        score = classifier.score(x_test, y_test)
        print(f"{len(sentences)} sentences, train shape: {x_train.shape}, score: {score}")
        model = models.Sequential()
        model.add(layers.Dense(10, input_dim=x_train.shape[1], activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        print("Model Summary:")
        model.summary()
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        history = model.fit(
            x_train, 
            y_train, 
            epochs=25, 
            validation_data=(x_test, y_test)
        )
        plot_history(history)

if __name__ == "__main__":
    BagOfWords()
    SentimentAnalysis("https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip", "/tmp/sentiment_data.zip")