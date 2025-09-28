import argparse, csv, numpy, pickle
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
from numpy import genfromtxt
from tensorflow import keras
from keras import saving
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Layer, Dot
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
import tabulate
pd.set_option("display.precision", 1)
@saving.register_keras_serializable()
class L2_Normalize(Layer):
    def call(self, x, axis):
        return tf.linalg.l2_normalize(x, axis)
    
class ContentBasedFiltering():
    """
    UnSupervised Learning.
    Implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings.
    The goal of a collaborative filtering recommender system is to generate two vectors: 
    For each user, a 'parameter vector' that embodies the movie tastes of a user. 
    For each movie, a feature vector of the same size which embodies some description of the movie. 
    The dot product of the two vectors plus the bias term should produce an estimate of the rating the user might give to that movie.
    Existing ratings are provided in matrix form as shown. Y contains ratings; 0.5 to 5 inclusive in 0.5 steps. 0 if the movie has not been rated. R has a 1 where movies have been rated. 
    Movies are in rows, users in columns. Each user has a parameter vector w^{user} and bias. Each movie has a feature vector x^{movie}. These vectors are simultaneously learned by using the existing user/movie ratings as training data. For example, w^{(1)} dot x^{(1)} + b^{(1)} = 4.
    It is worth noting that the feature vector $x^{movie} must satisfy all the users while the user vector w^{user} must satisfy all the movies. This is the source of the name of this approach - all the users collaborate to generate the rating set.
    Once the feature vectors and parameters are learned, they can be used to predict how a user might rate an unrated movie. This is shown in the diagram above. The equation is an example of predicting a rating for user one on movie zero.
    The function `cofiCostFunc` that computes the collaborative filtering objective function. After implementing the objective function, you will use a TensorFlow custom training loop to learn the parameters for collaborative filtering. The first step is to detail the data set and data structures that will be used in the lab.
    """
    _item_train = None
    _item_test = None
    _user_train = None
    _user_train_unscaled = None
    _user_test = None
    _y_train = None
    _y_test = None
    _item_NN = None
    _user_NN = None
    _item_features = None
    _user_features = None
    _item_vecs = None
    _movie_dict = None
    _user_to_genre = None
    _uvs = None
    _ivs = None
    _u_s = None
    _i_s = None
    _learning_rate: float = None
    _model = None
    _model_path = None
    _scalerUser: StandardScaler = None
    _scalerItem: StandardScaler = None
    _scalerTarget: MinMaxScaler = None
    def __init__(self, path:str, learning_rate:float):
        self._learning_rate = learning_rate
        self._model_path = path
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._model = tf.keras.models.load_model(self._model_path)
        self._prepare_data()
        self._scale_data()
        self._split_data()

    def _load_data(self):
        ''' called to load preprepared data for the lab '''
        self._item_train = genfromtxt('./data/ContentBasedFiltering/content_item_train.csv', delimiter=',')
        self._user_train = genfromtxt('./data/ContentBasedFiltering/content_user_train.csv', delimiter=',')
        self._y_train    = genfromtxt('./data/ContentBasedFiltering/content_y_train.csv', delimiter=',')
        with open('./data/ContentBasedFiltering/content_item_train_header.txt', newline='') as f:    #csv reader handles quoted strings better
            self._item_features = list(csv.reader(f))[0]
        with open('./data/ContentBasedFiltering/content_user_train_header.txt', newline='') as f:
            self._user_features = list(csv.reader(f))[0]
        self._item_vecs = genfromtxt('./data/ContentBasedFiltering/content_item_vecs.csv', delimiter=',')

        self._movie_dict = defaultdict(dict)
        count = 0
    #    with open('./data/ContentBasedFiltering/movies.csv', newline='') as csvfile:
        with open('./data/ContentBasedFiltering/content_movie_list.csv', newline='') as csvfile:
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

        with open('./data/ContentBasedFiltering/content_user_to_genre.pickle', 'rb') as f:
            self._user_to_genre = pickle.load(f)

    def _prepare_data(self):
        # Load Data, set configuration variables
        self._load_data()
        self._num_user_features = self._user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
        self._num_item_features = self._item_train.shape[1] - 1  # remove movie id at train time
        self._uvs = 3  # user genre vector start
        self._ivs = 3  # item genre vector start
        self._u_s = 3  # start of columns to use in training, user
        self._i_s = 1  # start of columns to use in training, items
        print(f"Number of training vectors: {len(self._item_train)}")

    def _scale_data(self):
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
    
    def _split_data(self):
        self._item_train, self._item_test = train_test_split(self._item_train, train_size=0.80, shuffle=True, random_state=1)
        self._user_train, self._user_test = train_test_split(self._user_train, train_size=0.80, shuffle=True, random_state=1)
        self._y_train, self._y_test       = train_test_split(self._y_train,    train_size=0.80, shuffle=True, random_state=1)
        print(f"movie/item training data shape: {self._item_train.shape}")
        print(f"movie/item test data shape: {self._item_test.shape}")
        self._pprint_train(self._user_train, self._user_features, self._uvs, self._u_s, maxcount=5)

    def BuildModel(self, rebuild: bool = False):
        if self._model and not rebuild:
            return
        num_outputs = 32
        tf.random.set_seed(1)
        self._user_NN = tf.keras.models.Sequential([
            Dense(256, activation = 'relu', kernel_regularizer=l2(0.1)), # Densely connected, or fully connected
            Dense(128, activation = 'relu', kernel_regularizer=l2(0.1)),
            Dense(num_outputs, activation = 'linear')
        ])

        self._item_NN = tf.keras.models.Sequential([
            Dense(256, activation = 'relu', kernel_regularizer=l2(0.1)),
            Dense(128, activation = 'relu', kernel_regularizer=l2(0.1)),
            Dense(num_outputs, activation = 'linear')
        ])

        # create the user input and point to the base network
        input_user = Input(shape=(self._num_user_features,))
        vu = self._user_NN(input_user)
        vu = L2_Normalize()(vu, axis=1)

        # create the item input and point to the base network
        input_item = Input(shape=(self._num_item_features,))
        vm = self._item_NN(input_item)
        vm = L2_Normalize()(vm, axis=1)

        # compute the dot product of the two vectors vu and vm
        output = Dot(axes=1)([vu, vm])

        # specify the inputs and output of the model
        self._model = tf.keras.Model([input_user, input_item], output)
        self._model.summary()
        self._train_model()
        self._evaluate_model()

    def _train_model(self):
        tf.random.set_seed(1)
        cost_fn = MeanSquaredError()
        opt = keras.optimizers.Adam(learning_rate=self._learning_rate) # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
        self._model.compile(optimizer=opt, loss=cost_fn)        
        self._model.fit([self._user_train[:, self._u_s:], self._item_train[:, self._i_s:]], self._y_train, epochs=30)
        if self._model_path:
            self._model.save(self._model_path)
            print(f"Model saved to {self._model_path}.")

    def _evaluate_model(self):
        self._model.evaluate([self._user_test[:, self._u_s:], self._item_test[:, self._i_s:]], self._y_test)

    def PredictNewUser(self, user):
        print(f"\n=== {self.PredictNewUser.__name__} ===")
        user_vec = numpy.array([[user["id"], user["rating_count"], user["rating_ave"],
                            user["action"], user["adventure"], user["animation"], user["childrens"],
                            user["comedy"], user["crime"], user["documentary"],
                            user["drama"], user["fantasy"], user["horror"], user["mystery"],
                            user["romance"], user["scifi"], user["thriller"]]])
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

        result = self._print_pred_movies(sorted_ypu, sorted_items, self._movie_dict, maxcount = 10)
        print(f"New user: {user["id"]}, recommendation:{result}")

    def PredictExistingUser(self):
        print(f"\n=== {self.PredictExistingUser.__name__} ===")
        uid = 2 
        # form a set of user vectors. This is the same vector, transformed and repeated.
        user_vecs, y_vecs = self._get_user_vecs(uid)

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
        result = self._print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, self._ivs, self._uvs, self._movie_dict, maxcount = 50)
        print(f"Existing user: {uid}, recommendation:{result}")

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

    def _get_user_vecs(self, user_id):
        """ 
        given a user_id, return:
            user train/predict matrix to match the size of item_vecs
            y vector with ratings for all rated movies and 0 for others of size item_vecs 
        self._user_train_unscaled, self._item_vecs, self._user_to_genre
        """
        if not user_id in self._user_to_genre:
            print("error: unknown user id")
            return None
        else:
            user_vec_found = False
            for i in range(len(self._user_train_unscaled)):
                if self._user_train_unscaled[i, 0] == user_id:
                    user_vec = self._user_train_unscaled[i]
                    user_vec_found = True
                    break
            if not user_vec_found:
                print("error in get_user_vecs, did not find uid in user_train")
            num_items = len(self._item_vecs)
            user_vecs = numpy.tile(user_vec, (num_items, 1))

            y = numpy.zeros(num_items)
            for i in range(num_items):  # walk through movies in item_vecs and get the movies, see if user has rated them
                movie_id = self._item_vecs[i, 0]
                if movie_id in self._user_to_genre[user_id]['movies']:
                    rating = self._user_to_genre[user_id]['movies'][movie_id]
                else:
                    rating = 0
                y[i] = rating
        return(user_vecs, y)

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

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Content-based filtering recommendation system')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    contentBasedFiltering = ContentBasedFiltering("models/ContentBasedFiltering.keras", 0.01)
    contentBasedFiltering.BuildModel(args.retrain)
    contentBasedFiltering.PredictExistingUser()
    user = {
        "id": 1234,
        "rating_ave": 0.0,
        "action": 0.0,
        "adventure": 5.0,
        "animation": 0.0,
        "childrens": 0.0,
        "comedy": 0.0,
        "crime": 0.0,
        "documentary": 0.0,
        "drama": 0.0,
        "fantasy": 5.0,
        "horror": 0.0,
        "mystery": 0.0,
        "romance": 0.0,
        "scifi": 0.0,
        "thriller": 0.0,
        "rating_count": 3

    }
    contentBasedFiltering.PredictNewUser(user)