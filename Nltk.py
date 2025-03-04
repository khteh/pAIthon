import nltk
from random import shuffle
from statistics import mean
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download([
     "names",
     "stopwords",
     "state_union",
     "twitter_samples",
     "movie_reviews",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt",
])
words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
stopwords = nltk.corpus.stopwords.words("english")

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
    print(f"=== {ConcordanceCollocations.__name__} ===")
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
    print(f"=== {Vader.__name__} ===")
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
    print(f"Movie review VADER evaluation: {correct / len(movie_review_ids): .2%} correct")

if __name__ == "__main__":
    FrequencyDistributions(nltk.corpus.state_union.raw())
    ConcordanceCollocations()
    Vader()
