import nltk
from pprint import pprint
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

if __name__ == "__main__":
    FrequencyDistributions(nltk.corpus.state_union.raw())
    ConcordanceCollocations()
