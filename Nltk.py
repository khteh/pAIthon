import nltk
from random import shuffle
from statistics import mean
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.naive_bayes import BernoulliNB,ComplementNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# https://realpython.com/python-nltk-sentiment-analysis/
nltk.download([
    "averaged_perceptron_tagger",
    "names",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "vader_lexicon",
    "punkt",
    "wordnet",
    "words"
])
stopwords = set(nltk.corpus.stopwords.words("english")) # includes only lowercase versions of stop words.
unwanted = stopwords
unwanted.update(set(nltk.corpus.names.words()))

def Tokenization():
    """
    (1) Tokenizing by word
    (2) Tokenizing by sentence
    """
    print(f"=== {Tokenization.__name__} ===")
    text = """
            Muad'Dib learned rapidly because his first training was in how to learn.
            And the first lesson of all was the basic trust that he could learn.
            It's shocking to find how many people do not believe they can learn,
            and how many more believe learning to be difficult."""
    words = word_tokenize(text)
    words_no_stop = [w for w in words if w.casefold() not in stopwords]
    sentences = sent_tokenize(text)
    print(f"Tokenize by word: {words[:10]}, {words_no_stop[:10]}")
    print(f"Tokenize by sentence: {sentences}")

def Stemming():
    """
    Stemming is a text processing task in which you reduce words to their root, which is the core part of a word. 
    For example, the words “helping” and “helper” share the root “help.” Stemming allows you to zero in on the basic 
    meaning of a word rather than all the details of how it’s being used.
    """
    print(f"\n=== {Stemming.__name__} ===")
    stemmer = PorterStemmer()
    text = """
            The crew of the USS Discovery discovered many discoveries.
            Discovering is what explorers do."""
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(w) for w in words]
    print(f"words: {words}")
    print(f"stemmed words: {stemmed_words}")

def PartsOfSpeechTagging():
    print(f"\n=== {PartsOfSpeechTagging.__name__} ===")
    text = """
            If you wish to make an apple pie from scratch,
            you must first invent the universe."""
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    print(f"pos_tags: {pos_tags}")

def Lemmatizing():
    """
    Like stemming, lemmatizing reduces words to their core meaning, but it will give you a complete English word that makes 
    sense on its own instead of just a fragment of a word like 'discoveri'.
    """
    print(f"\n=== {Lemmatizing.__name__} ===")
    text = """
            The crew of the USS Discovery discovered many discoveries.
            Discovering is what explorers do."""
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    print(f"words: {words}")
    print(f"lemmas: {lemmas}")
    text = "The friends of DeSoto love scarves."
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    print(f"words: {words}")
    print(f"lemmas: {lemmas}")
    print(f"Lemmatize 'worst': {lemmatizer.lemmatize('worst')}, as adjective: {lemmatizer.lemmatize('worst', pos='a')}")

def Chunking():
    """
    While tokenizing allows you to identify words and sentences, chunking allows you to identify phrases.
    """
    print(f"\n=== {Chunking.__name__} ===")
    text = "It's a dangerous business, Frodo, going out your door."
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    """
    Create a chunk grammar with one regular expression rule:
    Start with an optional (?) determiner ('DT')
    Can have any number (*) of adjectives (JJ)
    End with a noun (<NN>)    
    """
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    parser = nltk.RegexpParser(grammar)
    chunks = parser.parse(pos_tags)
    print(f"chunks: {chunks}")
    #chunks.draw() #Blocks

def Chinking():
    """
    Chinking is used together with chunking, but while chunking is used to include a pattern, chinking is used to exclude a pattern.
    """
    print(f"\n=== {Chinking.__name__} ===")
    text = "It's a dangerous business, Frodo, going out your door."
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    """
    The next step is to create a grammar to determine what you want to include and exclude in your chunks. 
    This time, you’re going to use more than one line because you’re going to have more than one rule.
    The first rule of your grammar is {<.*>+}. This rule has curly braces that face inward ({}) because it’s used to determine what patterns you want to include in you chunks. In this case, you want to include everything: <.*>+.
    The second rule of your grammar is }<JJ>{. This rule has curly braces that face outward (}{) because it’s used to determine what patterns you want to exclude in your chunks. In this case, you want to exclude adjectives: <JJ>.    
    """
    grammar = """
            Chunk: {<.*>+}
                    }<JJ>{
            """
    parser = nltk.RegexpParser(grammar)
    chunks = parser.parse(pos_tags)
    print(f"chunks: {chunks}")
    #chunks.draw() #Blocks

def NamedEntityRecognition():
    """
    Named entities are noun phrases that refer to specific locations, people, organizations, and so on. 
    With named entity recognition, you can find the named entities in your texts and also determine what kind of named entity they are.
    """
    print(f"\n=== {NamedEntityRecognition.__name__} ===")
    text = "It's a dangerous business, Frodo, going out your door."
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    chunks = nltk.ne_chunk(pos_tags)
    #chunks.draw() #Blocks
    text = """
        Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
        for countless centuries Mars has been the star of war—but failed to
        interpret the fluctuating appearances of the markings they mapped so well.
        All that time the Martians must have been getting ready.

        During the opposition of 1894 a great light was seen on the illuminated
        part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
        and then by other observers. English readers heard of it first in the
        issue of Nature dated August 2."""
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    chunks = nltk.ne_chunk(pos_tags, binary=True) # Just need to know what the names entities are but NOT their types
    named_entities = set(" ".join(i[0] for i in t) for t in chunks if hasattr(t, "label") and t.label() == "NE")
    print(f"Named entities: {named_entities}")

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
    frequencies_lower.plot(20, cumulative=True)
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
    print("\nconcordance list for 'america':")
    for c in concordances:
        print(c.line)
    words: list[str] = [w for w in nltk.word_tokenize(state_union_text) if w.isalpha() and w not in stopwords]
    bigrams = nltk.collocations.BigramCollocationFinder.from_words(words)
    print(f"\n10 Most common Bigrams: {bigrams.ngram_fd.most_common(10)}")
    print("Tabulated (10 Bigrams):")
    bigrams.ngram_fd.tabulate(10)

    trigrams = nltk.collocations.TrigramCollocationFinder.from_words(words)
    print(f"\n10 Most common Trigrams: {trigrams.ngram_fd.most_common(10)}")
    print("Tabulated (10 Trigrams):")
    trigrams.ngram_fd.tabulate(10)

    quadgrams = nltk.collocations.QuadgramCollocationFinder.from_words(words)
    print(f"\n10 Most common Quadgrams: {quadgrams.ngram_fd.most_common(10)}")
    print("Tabulated (10 Quadgrams):")
    quadgrams.ngram_fd.tabulate(10)

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    print("\n--- After lemmatizing the words... ---")
    text = nltk.Text(lemmas)
    text.concordance("america", lines=10)
    concordances = text.concordance_list("america", lines=10)
    print("\nconcordance list for 'america':")
    for c in concordances:
        print(c.line)
    words: list[str] = [w for w in nltk.word_tokenize(state_union_text) if w.isalpha() and w not in stopwords]

    bigrams = nltk.collocations.BigramCollocationFinder.from_words(words)
    print(f"\n10 Most common Bigrams: {bigrams.ngram_fd.most_common(10)}")
    print("Tabulated (10 Bigrams):")
    bigrams.ngram_fd.tabulate(10)

    trigrams = nltk.collocations.TrigramCollocationFinder.from_words(words)
    print(f"\n10 Most common Trigrams: {trigrams.ngram_fd.most_common(10)}")
    print("Tabulated (10 Trigrams):")
    trigrams.ngram_fd.tabulate(10)

    quadgrams = nltk.collocations.QuadgramCollocationFinder.from_words(words)
    print(f"\n10 Most common Quadgrams: {quadgrams.ngram_fd.most_common(10)}")
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

def BuildFeatures():
    print(f"\n=== {BuildFeatures.__name__} ===")
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
    shuffle(features)   
    return (features, top_100_positive)

def CustomizeSentimentAnalysis(features, top_100_positive):
    print(f"\n=== {CustomizeSentimentAnalysis.__name__} ===")
    # Use 1/4 of the set for training
    train_count = len(features) // 4 # Integer division
    print("\nfeatures:")
    print(features[:10])
    print(f"\nTraining the NaiveBayesClassifier with {train_count} features...")
    classifier = nltk.NaiveBayesClassifier.train(features[:train_count])
    classifier.show_most_informative_features(10)
    print(f"\nCustom sentiment analysis accuracy: {nltk.classify.accuracy(classifier, features[train_count:])}")
    feature = ExtractCustomFeatures("To be or not to be", top_100_positive)
    print("\nfeature:")
    print(feature)
    result = classifier.classify(feature)
    print(f"\nClassification result: {result}")
    classifier.show_most_informative_features()

def SentimentAnalysisUsingScikitLearnClassifiers(features):
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
    train_count = len(features) // 4 # Integer division
    for name, classifier in classifiers.items():
        c = nltk.classify.SklearnClassifier(classifier)
        c.train(features[:train_count])
        accuracy = nltk.classify.accuracy(c, features[train_count:])
        print(f"{name}: {accuracy:.2%}")

if __name__ == "__main__":
    Tokenization()
    Stemming()
    PartsOfSpeechTagging()
    Lemmatizing()
    Chunking()
    Chinking()
    NamedEntityRecognition()
    FrequencyDistributions(nltk.corpus.state_union.raw())
    ConcordanceCollocations()
    Vader()
    features, top_100_positive = BuildFeatures()
    CustomizeSentimentAnalysis(features, top_100_positive)
    SentimentAnalysisUsingScikitLearnClassifiers(features)"