import nltk
from random import shuffle
from statistics import mean
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import BernoulliNB,ComplementNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
nltk.download([
     "names",
     "stopwords",
     "state_union",
     "twitter_samples",
     "movie_reviews",
     "averaged_perceptron_tagger",
     "averaged_perceptron_tagger_eng",
     "vader_lexicon",
     "punkt",
])
stopwords = nltk.corpus.stopwords.words("english")
unwanted = stopwords
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

def FrequencyDistributions(corpus: str):
    """
    Create words frequencies using:
    (1) nltk.FreqDist
    (2) nltk.Text().vocab()
    """
    print(f"=== {FrequencyDistributions.__name__} ===")
    words: list[str] = [w for w in nltk.word_tokenize(corpus) if w.isalpha() and w not in stopwords]
    #print("Tokens / words:")
    #pprint(words, width=79, compact=True)
    frequencies = nltk.FreqDist(words)
    print(f"10 Most common: {frequencies.most_common(10)}")
    print("Tabulated (10):")
    frequencies.tabulate(10)
    frequencies_lower = nltk.FreqDist([w.lower() for w in frequencies])
    print(f"\n10 Most common (lower-cased): {frequencies_lower.most_common(10)}")
    print("\nTabulated (10 lower-cased):")
    frequencies_lower.tabulate(10)
    text = nltk.Text(words)
    frequencies_text = text.vocab() # Equivalent to fd = nltk.FreqDist(words)
    print(f"\n10 Most common (nltk.Text): {frequencies_text.most_common(10)}")
    print("Tabulated (10 nltk.Text):")
    frequencies_text.tabulate(10)

def ConcordanceCollocations():
    """
    In the context of NLP, a concordance is a collection of word locations along with their context. 
    """
    print(f"\n=== {ConcordanceCollocations.__name__} ===")
    state_union_words: list[str] = nltk.corpus.state_union.words()
    state_union_text: str = nltk.corpus.state_union.raw()
    text = nltk.Text(state_union_words)
    text.concordance("america", lines=10)
    concordances = text.concordance_list("america", lines=10)
    print("\nconcordance list:")
    for c in concordances:
        print(c.line)
    words: list[str] = [w for w in nltk.word_tokenize(state_union_text) if w.isalpha() and w not in stopwords]
    bigrams = nltk.collocations.BigramCollocationFinder.from_words(words)
    print(f"\n10 Most common Bigrams: {bigrams.ngram_fd.most_common(10)}")
    print("Tabulated (10 Bigrams):")
    bigrams.ngram_fd.tabulate(10)

    trigrams = nltk.collocations.TrigramCollocationFinder.from_words(words)
    print(f"10 Most common Trigrams: {trigrams.ngram_fd.most_common(10)}")
    print("Tabulated (10 Trigrams):")
    trigrams.ngram_fd.tabulate(10)

    quadgrams = nltk.collocations.QuadgramCollocationFinder.from_words(words)
    print(f"10 Most common Quadgrams: {quadgrams.ngram_fd.most_common(10)}")
    print("Tabulated (10 Quadgrams):")
    quadgrams.ngram_fd.tabulate(10)

def Vader():
    """
    NLTK already has a built-in, pretrained sentiment analyzer called VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Since VADER is pretrained, you can get results more quickly than with many other analyzers. However, VADER is best suited for language used in social media, like short sentences with some slang and abbreviations. It’s less accurate when rating longer, structured sentences, but it’s often a good launching point.
    """
    print(f"\n=== {Vader.__name__} ===")
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores("Wow, NLTK is cool!")
    print(f"Score: {score}")
    tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]
    shuffle(tweets)
    print(f"\nTweets sentiment analysis:")
    for t in tweets[:10]:
        sentiment = analyzer.polarity_scores(t)["compound"] > 0
        print(f"{t} => {'Positive' if sentiment else 'Negative'}")
    print(f"\nMovie reviews sentiment analysis:")
    positive_review_ids = nltk.corpus.movie_reviews.fileids(categories="pos")
    negative_review_ids = nltk.corpus.movie_reviews.fileids(categories="neg")
    movie_review_ids = positive_review_ids + negative_review_ids
    shuffle(movie_review_ids)
    correct = 0
    for id in movie_review_ids:
        review = nltk.corpus.movie_reviews.raw(id)
        # nltk.sent_tokenize() to obtain a list of sentences from the review text
        scores = [
            analyzer.polarity_scores(sentence)["compound"] for sentence in nltk.sent_tokenize(review)
        ]
        score = mean(scores)
        if score > 0:
            if id in positive_review_ids:
                correct += 1
        else:
            if id in negative_review_ids:
                correct += 1
    print(f"Movie reviews VADER evaluation: {correct / len(movie_review_ids): .2%} correct")

def SkipUnwanted(tuple) -> bool:
    word, tag = tuple
    return word.isalpha() and word not in unwanted and not tag.startswith("NN")

def ExtractCustomFeatures(text:str, top_100_positive):
    analyzer = SentimentIntensityAnalyzer()
    features = dict()
    positives = 0
    compound_scores = list()
    positive_scores = list()
    # nltk.sent_tokenize() to obtain a list of sentences from the review text
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100_positive:
                positives += 1
        compound_scores.append(analyzer.polarity_scores(sentence)["compound"])
        positive_scores.append(analyzer.polarity_scores(sentence)["pos"])
    # Add 1 to the final compound score so as to always have positive numbers
    # Some of the classifiers used later in the process do not work with negative numbers
    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    features["positives"] = positives
    return features

def CustomizeSentimentAnalysis():
    print(f"\n=== {CustomizeSentimentAnalysis.__name__} ===")
    positive_words = nltk.corpus.movie_reviews.words(categories=["pos"])
    negative_words = nltk.corpus.movie_reviews.words(categories=["neg"])
    positive = [word for word, tag in filter(
        SkipUnwanted, 
        nltk.pos_tag(positive_words))]
    negative = [word for word, tag in filter(
        SkipUnwanted, 
        nltk.pos_tag(negative_words))]
    print(f"positive words: {positive[:10]}")
    print(f"negatives words: {negative[:10]}")
    positive_freq = nltk.FreqDist(positive)
    negative_freq = nltk.FreqDist(negative)
    print(f"10 positives: {positive_freq.most_common(10)}")
    print("Tabulated (10) positives:")
    positive_freq.tabulate(10)
    print(f"10 negatives: {negative_freq.most_common(10)}")
    print("Tabulated (10) negatives:")
    negative_freq.tabulate(10)
    common = set(positive_freq).intersection(negative_freq)
    for w in common:
        del positive_freq[w]
        del negative_freq[w]
    top_100_positive = {w for w, count in positive_freq.most_common(100)}
    top_100_negative = {w for w, count in negative_freq.most_common(100)}
    positive_bigrams = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in positive_words if w.isalpha() and w not in unwanted])
    negative_bigrams = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in negative_words if w.isalpha() and w not in unwanted])
    features = [(ExtractCustomFeatures(nltk.corpus.movie_reviews.raw(review), top_100_positive), "pos")
                for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])]
    features.extend([(ExtractCustomFeatures(nltk.corpus.movie_reviews.raw(review), top_100_positive), "neg")
                for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])])
    # Use 1/4 of the set for training
    train_count = len(features) // 4 # Integer division
    shuffle(features)
    print(f"Training the NaiveBayesClassifier with {train_count} features...")
    classifier = nltk.NaiveBayesClassifier.train(features[:train_count])
    classifier.show_most_informative_features(10)
    print(f"Custom sentiment analysis accuracy: {nltk.classify.accuracy(classifier, features[train_count:])}")
    review = "To be or not to be"
    result = classifier.classify(dict(review, True))
    print(f"{review} classification result: {result}")
    print(ExtractCustomFeatures(review, top_100_positive))

def SentimentAnalysisUsingScikitLearnClassifiers():
    print(f"\n=== {SentimentAnalysisUsingScikitLearnClassifiers.__name__} ===")
    classifiers = {
        "BernoulliNB": BernoulliNB(),
        "ComplementNB": ComplementNB(),
        "MultinomialNB": MultinomialNB(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(),
        "MLPClassifier": MLPClassifier(max_iter=1000),
        "AdaBoostClassifier": AdaBoostClassifier(),
    }
    positive_words = nltk.corpus.movie_reviews.words(categories=["pos"])
    negative_words = nltk.corpus.movie_reviews.words(categories=["neg"])
    positive = [word for word, tag in filter(
        SkipUnwanted, 
        nltk.pos_tag(positive_words))]
    negative = [word for word, tag in filter(
        SkipUnwanted, 
        nltk.pos_tag(negative_words))]
    print(f"positive words: {positive[:10]}")
    print(f"negatives words: {negative[:10]}")
    positive_freq = nltk.FreqDist(positive)
    negative_freq = nltk.FreqDist(negative)
    print(f"10 positives: {positive_freq.most_common(10)}")
    print("Tabulated (10) positives:")
    positive_freq.tabulate(10)
    print(f"10 negatives: {negative_freq.most_common(10)}")
    print("Tabulated (10) negatives:")
    negative_freq.tabulate(10)
    common = set(positive_freq).intersection(negative_freq)
    for w in common:
        del positive_freq[w]
        del negative_freq[w]
    top_100_positive = {w for w, count in positive_freq.most_common(100)}
    top_100_negative = {w for w, count in negative_freq.most_common(100)}
    positive_bigrams = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in positive_words if w.isalpha() and w not in unwanted])
    negative_bigrams = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in negative_words if w.isalpha() and w not in unwanted])
    features = [(ExtractCustomFeatures(nltk.corpus.movie_reviews.raw(review), top_100_positive), "pos")
                for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])]
    features.extend([(ExtractCustomFeatures(nltk.corpus.movie_reviews.raw(review), top_100_positive), "neg")
                for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])])

if __name__ == "__main__":
    FrequencyDistributions(nltk.corpus.state_union.raw())
    ConcordanceCollocations()
    Vader()
    CustomizeSentimentAnalysis()
    SentimentAnalysisUsingScikitLearnClassifiers()