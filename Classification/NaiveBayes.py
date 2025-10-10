import numpy, pandas, nltk, string, pdb, re
from nltk.corpus import stopwords, twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from os import getcwd

nltk.download('twitter_samples')
nltk.download('stopwords')

filePath = f"{getcwd()}/../data/"
nltk.data.path.append(filePath)

def PrepareData():
    print(f"\n=== {PrepareData.__name__} ===")
    # get the sets of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    # split the data into two pieces, one for training and one for testing (validation set)
    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]

    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    # avoid assumptions about the length of all_positive_tweets
    train_y = numpy.append(numpy.ones(len(train_pos)), numpy.zeros(len(train_neg)))
    test_y = numpy.append(numpy.ones(len(test_pos)), numpy.zeros(len(test_neg)))
    return train_x, train_y, test_x, test_y

def process_tweet(tweet):
    '''
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    '''
    #print(f"\n=== {process_tweet.__name__} ===")
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    #tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    print(f"\n=== {count_tweets.__name__} ===")
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word, y)
            
            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
    return result

def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels corresponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    print(f"\n=== {train_naive_bayes.__name__} ===")
    loglikelihood = {}
    logprior = 0

    # calculate V, the number of unique words in the vocabulary
    vocab = set([key[0] for key in freqs.keys()])
    V = len(vocab)

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]
    
    # Calculate D, the number of documents
    D = len(train_y)

    # Calculate D_pos, the number of positive documents
    D_pos = (train_y == 1).sum()
    P_D_pos = D_pos / len(train_y)

    # Calculate D_neg, the number of negative documents
    D_neg = (train_y == 0).sum()
    P_D_neg = D_neg / len(train_y)

    # Calculate logprior
    # Prior = P(pos) / P(neg) = P_D_pos / P_D_neg
    logprior = numpy.log(D_pos) - numpy.log(D_neg)
    
    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs[(word, 1)] if (word, 1) in freqs else 0
        freq_neg = freqs[(word, 0)] if (word, 0) in freqs else 0

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = numpy.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood

def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    #print(f"\n=== {naive_bayes_predict.__name__} ===")
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0.0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p

def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    """
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    print(f"\n=== {test_naive_bayes.__name__} ===")
    accuracy = 0  # return this properly
    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = numpy.abs(numpy.subtract(y_hats, test_y)).sum() / len(test_y)

    # Accuracy is 1 minus the error
    accuracy = 1 - error
    return accuracy

def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    ### START CODE HERE ###
    # use lookup() to find positive counts for the word (denoted by the integer 1)
    pos_neg_ratio['positive'] = freqs[(word, 1)] if (word, 1) in freqs else 0
    
    # use lookup() to find negative counts for the word (denoted by integer 0)
    pos_neg_ratio['negative'] = freqs[(word, 0)] if (word, 0) in freqs else 0
    
    # calculate the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)
    ### END CODE HERE ###
    return pos_neg_ratio

def get_words_by_threshold(freqs, label, threshold):
    '''
    If we set the label to 1, then we'll look for all words whose threshold of positive/negative is at least as high as that threshold, or higher.
    If we set the label to 0, then we'll look for all words whose threshold of positive/negative is at most as low as the given threshold, or lower.

    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_list: dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
        example of a key value pair:
        {'happi':
            {'positive': 10, 'negative': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}
    for key in freqs.keys():
        word, _ = key

        # get the positive/negative ratio for a word
        pos_neg_ratio = get_ratio(freqs, word)

        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:
        
            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
        
            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # otherwise, do not include this word in the list (do nothing)

    return word_list

def SentimentAnalysisTests(logprior, loglikelihood):
    print(f"\n=== {SentimentAnalysisTests.__name__} ===")
    for tweet in ['She smiled.', 'He laughed.', 'I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great', 'you are bad :(']:
        # print( '%s -> %f' % (tweet, naive_bayes_predict(tweet, logprior, loglikelihood)))
        print(f"{tweet} : {naive_bayes_predict(tweet, logprior, loglikelihood):.2f}")

def get_words_by_threshold_tests(freqs):
    print(f"\n=== {get_words_by_threshold_tests.__name__} ===")
    # Test your function: find negative words at or below a threshold
    print(f"label: 0, threshold: 0.05")
    print(get_words_by_threshold(freqs, label=0, threshold=0.05))
    print(f"\nlabel: 1, threshold: 10")
    print(get_words_by_threshold(freqs, label=1, threshold=10))

def ErrorAnalysis():
    print(f"\n=== {ErrorAnalysis.__name__} ===")
    print('Truth Predicted Tweet')
    for x, y in zip(test_x, test_y):
        y_hat = naive_bayes_predict(x, logprior, loglikelihood)
        if y != (numpy.sign(y_hat) > 0):
            print('%d\t%0.2f\t%s' % (y, numpy.sign(y_hat) > 0, ' '.join(
                process_tweet(x)).encode('ascii', 'ignore')))
            
if __name__ == "__main__":
    custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"

    train_x, train_y, test_x, test_y = PrepareData()
    # print cleaned tweet
    print(process_tweet(custom_tweet))

    result = {}
    tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
    ys = [1, 0, 0, 0, 0]
    count_tweets(result, tweets, ys)

    # Build the freqs dictionary for later uses
    freqs = count_tweets({}, train_x, train_y)
    logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
    print(f"logprior: {logprior}, loglikelihood: {len(loglikelihood)}")

    print(f"Naive Bayes accuracy = {test_naive_bayes(test_x, test_y, logprior, loglikelihood):.4f}")

    SentimentAnalysisTests(logprior, loglikelihood)
    ErrorAnalysis()
    get_words_by_threshold_tests(freqs)
