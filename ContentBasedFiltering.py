import csv, numpy
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
import pickle5 as pickle
from collections import defaultdict
from numpy import genfromtxt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, losses, optimizers, regularizers
import tabulate
from recsysNN_utils import *
pd.set_option("display.precision", 1)

class ContentBasedFiltering():
    _item_train = None
    _item_test = None
    _user_train = None
    _user_train_unscaled = None
    _user_test = None
    _y_train = None
    _y_test = None
    _item_NN = None
    _item_features = None
    _user_features = None
    _item_vecs = None
    _movie_dict = None
    _user_to_genre = None
    _uvs = None
    _u_s = None
    _i_s = None
    _model = None
    _scalerUser: StandardScaler = None
    _scalerItem: StandardScaler = None
    _scalerTarget: MinMaxScaler = None
    def __init__(self, path):
        self.PrepareData()
        self.ScaleData()
        self.SplitData()

    def _load_data(self):
        ''' called to load preprepared data for the lab '''
        self._item_train = genfromtxt('./data/content_item_train.csv', delimiter=',')
        self._user_train = genfromtxt('./data/content_user_train.csv', delimiter=',')
        self._y_train    = genfromtxt('./data/content_y_train.csv', delimiter=',')
        with open('./data/content_item_train_header.txt', newline='') as f:    #csv reader handles quoted strings better
            self._item_features = list(csv.reader(f))[0]
        with open('./data/content_user_train_header.txt', newline='') as f:
            self._user_features = list(csv.reader(f))[0]
        self._item_vecs = genfromtxt('./data/content_item_vecs.csv', delimiter=',')

        self._movie_dict = defaultdict(dict)
        count = 0
    #    with open('./data/movies.csv', newline='') as csvfile:
        with open('./data/content_movie_list.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for line in reader:
                if count == 0:
                    count += 1  #skip header
                    #print(line) print
                else:
                    count += 1
                    movie_id = int(line[0])
                    self._movie_dict[movie_id]["title"] = line[1]
                    self._movie_dict[movie_id]["genres"] = line[2]

        with open('./data/content_user_to_genre.pickle', 'rb') as f:
            self._user_to_genre = pickle.load(f)

    def PrepareData(self):
        # Load Data, set configuration variables
        self._load_data()
        self._num_user_features = self._user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
        self._num_item_features = self._item_train.shape[1] - 1  # remove movie id at train time
        self._uvs = 3  # user genre vector start
        ivs = 3  # item genre vector start
        self._u_s = 3  # start of columns to use in training, user
        self._i_s = 1  # start of columns to use in training, items
        print(f"Number of training vectors: {len(self._item_train)}")

    def ScaleData(self):
        # scale training data
        item_train_unscaled = self._item_train
        self._user_train_unscaled = self._user_train
        y_train_unscaled    = self._y_train

        self._scalerItem = StandardScaler()
        self._scalerItem.fit(self._item_train)
        self._item_train = self._scalerItem.transform(self._item_train)

        self._scalerUser = StandardScaler()
        self._scalerUser.fit(self._user_train)
        self._user_train = self._scalerUser.transform(self._user_train)

        self._scalerTarget = MinMaxScaler((-1, 1))
        self._scalerTarget.fit(self._y_train.reshape(-1, 1))
        self._y_train = self._scalerTarget.transform(self._y_train.reshape(-1, 1))
        #ynorm_test = self._scalerTarget.transform(y_test.reshape(-1, 1))

        print(numpy.allclose(item_train_unscaled, self._scalerItem.inverse_transform(self._item_train)))
        print(numpy.allclose(self._user_train_unscaled, self._scalerUser.inverse_transform(self._user_train)))
    
    def SplitData(self):
        self._item_train, self._item_test = train_test_split(self._item_train, train_size=0.80, shuffle=True, random_state=1)
        self._user_train, self._user_test = train_test_split(self._user_train, train_size=0.80, shuffle=True, random_state=1)
        self._y_train, self._y_test       = train_test_split(self._y_train,    train_size=0.80, shuffle=True, random_state=1)
        print(f"movie/item training data shape: {self._item_train.shape}")
        print(f"movie/item test data shape: {self._item_test.shape}")
        self._pprint_train(self._user_train, self._user_features, self._uvs, self._u_s, maxcount=5)

    def BuildModels(self):
        num_outputs = 32
        tf.random.set_seed(1)
        user_NN = tf.keras.models.Sequential([
            Dense(256, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(num_outputs, activation = 'linear')
        ])

        self._item_NN = tf.keras.models.Sequential([
            Dense(256, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(num_outputs, activation = 'linear')
        ])

        # create the user input and point to the base network
        input_user = tf.keras.layers.Input(shape=(self._num_user_features))
        vu = user_NN(input_user)
        vu = tf.linalg.l2_normalize(vu, axis=1)

        # create the item input and point to the base network
        input_item = tf.keras.layers.Input(shape=(self._num_item_features))
        vm = self._item_NN(input_item)
        vm = tf.linalg.l2_normalize(vm, axis=1)

        # compute the dot product of the two vectors vu and vm
        output = tf.keras.layers.Dot(axes=1)([vu, vm])

        # specify the inputs and output of the model
        self._model = tf.keras.Model([input_user, input_item], output)
        self._model.summary()

    def TrainModel(self):
        tf.random.set_seed(1)
        cost_fn = tf.keras.losses.MeanSquaredError()
        opt = keras.optimizers.Adam(learning_rate=0.01)
        self._model.compile(optimizer=opt, loss=cost_fn)        
        tf.random.set_seed(1)
        self._model.fit([self._user_train[:, self._u_s:], self._item_train[:, self._i_s:]], self._y_train, epochs=30)

    def EvaluateModel(self):
        self._model.evaluate([self._user_test[:, self._u_s:], self._item_test[:, self._i_s:]], self._y_test)

    def PredictNewUser(self):
        new_user_id = 5000
        new_rating_ave = 0.0
        new_action = 0.0
        new_adventure = 5.0
        new_animation = 0.0
        new_childrens = 0.0
        new_comedy = 0.0
        new_crime = 0.0
        new_documentary = 0.0
        new_drama = 0.0
        new_fantasy = 5.0
        new_horror = 0.0
        new_mystery = 0.0
        new_romance = 0.0
        new_scifi = 0.0
        new_thriller = 0.0
        new_rating_count = 3
        user_vec = numpy.array([[new_user_id, new_rating_count, new_rating_ave,
                            new_action, new_adventure, new_animation, new_childrens,
                            new_comedy, new_crime, new_documentary,
                            new_drama, new_fantasy, new_horror, new_mystery,
                            new_romance, new_scifi, new_thriller]])
        # generate and replicate the user vector to match the number movies in the data set.
        user_vecs = self._gen_user_vecs(user_vec,len(self._item_vecs))

        # scale our user and item vectors
        suser_vecs = self._scalerUser.transform(user_vecs)
        sitem_vecs = self._scalerItem.transform(self._item_vecs)

        # make a prediction
        y_p = self._model.predict([suser_vecs[:, self._u_s:], sitem_vecs[:, self._i_s:]])

        # unscale y prediction 
        y_pu = self._scalerTarget.inverse_transform(y_p)

        # sort the results, highest prediction first
        sorted_index = numpy.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
        sorted_ypu   = y_pu[sorted_index]
        sorted_items = self._item_vecs[sorted_index]  #using unscaled vectors for display

        self._print_pred_movies(sorted_ypu, sorted_items, self._movie_dict, maxcount = 10)

    def PredictExistingUser(self):
        uid = 2 
        # form a set of user vectors. This is the same vector, transformed and repeated.
        user_vecs, y_vecs = self._get_user_vecs(uid, self._user_train_unscaled, self._item_vecs, self._user_to_genre)

        # scale our user and item vectors
        suser_vecs = self._scalerUser.transform(user_vecs)
        sitem_vecs = self._scalerItem.transform(self._item_vecs)

        # make a prediction
        y_p = self._model.predict([suser_vecs[:, self._u_s:], sitem_vecs[:, self._i_s:]])

        # unscale y prediction 
        y_pu = self._scalerTarget.inverse_transform(y_p)

        # sort the results, highest prediction first
        sorted_index = numpy.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
        sorted_ypu   = y_pu[sorted_index]
        sorted_items = self._item_vecs[sorted_index]  #using unscaled vectors for display
        sorted_user  = user_vecs[sorted_index]
        sorted_y     = y_vecs[sorted_index]

        #print sorted predictions for movies rated by the user
        self._print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, self._ivs, self._uvs, self._movie_dict, maxcount = 50)        

    def FindSimilarItems(self):
        """
        A matrix of distances between movies can be computed once when the model is trained and then reused for new recommendations without retraining. The first step, once a model is trained, is to obtain the movie feature vector, Vm, for each of the movies. 
        To do this, we will use the trained item_NN and build a small model to allow us to run the movie vectors through it to generate Vm
        """
        input_item_m = tf.keras.layers.Input(shape=(self._num_item_features))    # input layer
        vm_m = self._item_NN(input_item_m)                                       # use the trained item_NN
        vm_m = tf.linalg.l2_normalize(vm_m, axis=1)                        # incorporate normalization as was done in the original model
        model_m = tf.keras.Model(input_item_m, vm_m)                                
        model_m.summary()

        # Once you have a movie model, you can create a set of movie feature vectors by using the model to predict using a set of item/movie vectors as input. item_vecs is a set of all of the movie vectors. It must be scaled to use with the trained model. 
        # The result of the prediction is a 32 entry feature vector for each movie.        
        scaled_item_vecs = self._scalerItem.transform(self._item_vecs)
        vms = model_m.predict(scaled_item_vecs[:,self._i_s:])
        print(f"size of all predicted movie feature vectors: {vms.shape}")

        # Let's now compute a matrix of the squared distance between each movie feature vector and all other movie feature vectors
        # We can then find the closest movie by finding the minimum along each row. We will make use of numpy masked arrays to avoid selecting the same movie. The masked values along the diagonal won't be included in the computation.
        count = 50  # number of movies to display
        dim = len(vms)
        dist = numpy.zeros((dim,dim))

        for i in range(dim):
            for j in range(dim):
                dist[i,j] = self._sq_dist(vms[i, :], vms[j, :])
                
        m_dist = ma.masked_array(dist, mask=numpy.identity(dist.shape[0]))  # mask the diagonal

        disp = [["movie1", "genres", "movie2", "genres"]]
        for i in range(count):
            min_idx = numpy.argmin(m_dist[i])
            movie1_id = int(self._item_vecs[i,0])
            movie2_id = int(self._item_vecs[min_idx,0])
            disp.append( [self._movie_dict[movie1_id]['title'], self._movie_dict[movie1_id]['genres'],
                        self._movie_dict[movie2_id]['title'], self._movie_dict[movie1_id]['genres']]
                    )
        table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
        print(table)

    def _sq_dist(self, a,b):
        """
        Returns the squared distance between two vectors
        Args:
        a (ndarray (n,)): vector with n features
        b (ndarray (n,)): vector with n features
        Returns:
        d (float) : distance
        """
        return numpy.sum(numpy.square(a-b))
    
    def _gen_user_vecs(self, user_vec, num_items):
        """ given a user vector return:
            user predict maxtrix to match the size of item_vecs """
        user_vecs = numpy.tile(user_vec, (num_items, 1))
        return user_vecs

    def _print_pred_movies(self, y_p, item, movie_dict, maxcount=10):
        """ print results of prediction of a new user. inputs are expected to be in
            sorted order, unscaled. """
        count = 0
        disp = [["y_p", "movie id", "rating ave", "title", "genres"]]
        for i in range(0, y_p.shape[0]):
            if count == maxcount:
                break
            count += 1
            movie_id = item[i, 0].astype(int)
            disp.append([numpy.around(y_p[i, 0], 1), item[i, 0].astype(int), numpy.around(item[i, 2].astype(float), 1),
                        movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

        table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
        return table
    
    def _print_existing_user(self, y_p, y, user, items, ivs, uvs, movie_dict, maxcount=10):
        """ print results of prediction for a user who was in the database.
            Inputs are expected to be in sorted order, unscaled.
        """
        count = 0
        disp = [["y_p", "y", "user", "user genre ave", "movie rating ave", "movie id", "title", "genres"]]
        count = 0
        for i in range(0, y.shape[0]):
            if y[i, 0] != 0:  # zero means not rated
                if count == maxcount:
                    break
                count += 1
                movie_id = items[i, 0].astype(int)

                offsets = numpy.nonzero(items[i, ivs:] == 1)[0]
                genre_ratings = user[i, uvs + offsets]
                disp.append([y_p[i, 0], y[i, 0],
                            user[i, 0].astype(int),      # userid
                            numpy.array2string(genre_ratings, 
                                            formatter={'float_kind':lambda x: "%.1f" % x},
                                            separator=',', suppress_small=True),
                            items[i, 2].astype(float),    # movie average rating
                            movie_id,
                            movie_dict[movie_id]['title'],
                            movie_dict[movie_id]['genres']])

        table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".1f"])
        return table

    def _pprint_train(self, x_train, features, vs, u_s, maxcount=5, user=True):
        """ Prints user_train or item_train nicely """
        if user:
            flist = [".0f", ".0f", ".1f",
                    ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f"]
        else:
            flist = [".0f", ".0f", ".1f", 
                    ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f"]

        head = features[:vs]
        if vs < u_s: print("error, vector start {vs} should be greater then user start {u_s}")
        for i in range(u_s):
            head[i] = "[" + head[i] + "]"
        genres = features[vs:]
        hdr = head + genres
        disp = [self._split_str(hdr, 5)]
        count = 0
        for i in range(0, x_train.shape[0]):
            if count == maxcount: break
            count += 1
            disp.append([x_train[i, 0].astype(int),
                        x_train[i, 1].astype(int),
                        x_train[i, 2].astype(float),
                        *x_train[i, 3:].astype(float)
                        ])
        table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=flist, numalign='center')
        return table
    def _split_str(self, ifeatures, smax):
        ''' split the feature name strings to tables fit '''
        ofeatures = []
        for s in ifeatures:
            if not ' ' in s:  # skip string that already have a space
                if len(s) > smax:
                    mid = int(len(s)/2)
                    s = s[:mid] + " " + s[mid:]
            ofeatures.append(s)
        return ofeatures
