import numpy, pandas as pd
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
"""
Collaborative filtering movie recommendation system.
The dataset is derived from https://grouplens.org/datasets/movielens/latest/
The original dataset has  9000 movies rated by 600 users. The dataset has been reduced in size to focus on movies from the years since 2000. This dataset consists of ratings on a scale of 0.5 to 5 in 0.5 step increments. The reduced dataset has $n_u = 443$ users, and $n_m= 4778$ movies.
"""
class MovieRecommendation():
    _data_path: str = None
    _num_movies: int = None
    _num_users:int = None
    _num_features: int = None
    _Y = None
    _R = None # Binary indicator matrix. R(i,j) = 1 if User-j rated Movie-i.
    _Ynorm = None
    _Ymean = None
    _W = None
    _X = None
    _b = None
    _lambda: float = None # Regularization param
    _optimizer = None
    _my_rated = None
    _movieList: list = None
    _movieList_df: pd.DataFrame = None
    _my_ratings: list = None
    def __init__(self, path, num_features:int, lambda_: float = 1):
        self._data_path = path
        self._num_features = num_features
        self._lambda = lambda_
        self.PrepareData()

    def load_Movie_List_pd(self):
        """ returns df with and index of movies in the order they are in in the Y matrix """
        df = pd.read_csv(self._data_path, header=0, index_col=0,  delimiter=',', quotechar='"')
        mlist = df["title"].to_list()
        return(mlist, df)
    
    def _load_ratings_small(self):
        file = open('./data/small_movies_Y.csv', 'rb')
        Y = loadtxt(file,delimiter = ",")

        file = open('./data/small_movies_R.csv', 'rb')
        R = loadtxt(file,delimiter = ",")
        return (Y,R)
    
    def _normalizeRatings(self):
        """
        Mean Normalization: Makes training faster and prediction of new row / column more reasonable instead of '0'. Without the mean of the rows/cols, 'w' ~= 0 (because of regularization) and therefore, 'z' is likely to be 0.
        Preprocess data by subtracting mean rating for every movie (every row).
        Only include real ratings R(i,j)=1.
        [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
        has a rating of 0 on average. Unrated moves then have a mean rating (0)
        Returns the mean rating in Ymean.
        """
        self._Ymean = (numpy.sum(self._Y * self._R, axis=1)/(numpy.sum(self._R, axis=1)+1e-12)).reshape(-1,1)
        self._Ynorm = self._Y - numpy.multiply(self._Ymean, self._R) 

    def _load_precalc_params_small(self):
        file = open('./data/small_movies_X.csv', 'rb')
        X = loadtxt(file, delimiter = ",")

        file = open('./data/small_movies_W.csv', 'rb')
        W = loadtxt(file,delimiter = ",")

        file = open('./data/small_movies_b.csv', 'rb')
        b = loadtxt(file,delimiter = ",")
        b = b.reshape(1,-1)
        self._num_movies, num_features = X.shape
        num_users,_ = W.shape
        return(X, W, b, num_features, num_users)

    def PrepareData(self):
        print(f"\n=== {self.PrepareData.__name__} ===")
        X, W, b, num_features, num_users = self._load_precalc_params_small()
        #Y, R = self._load_ratings_small()
        #print("Y", Y.shape, "R", R.shape)
        print("X", X.shape)
        print("W", W.shape)
        print("b", b.shape)
        print("num_features", num_features)
        print("num_movies",   self._num_movies)
        print("num_users",    num_users)
        self._movieList, self._movieList_df = self.load_Movie_List_pd()

        self._my_ratings = numpy.zeros(self._num_movies)          #  Initialize my ratings

        # Check the file small_movie_list.csv for id of each movie in our dataset
        # For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
        self._my_ratings[2700] = 5

        #Or suppose you did not enjoy Persuasion (2007), you can set
        self._my_ratings[2609] = 2;

        # We have selected a few movies we liked / did not like and the ratings we
        # gave are as follows:
        self._my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
        self._my_ratings[246]  = 5   # Shrek (2001)
        self._my_ratings[2716] = 3   # Inception
        self._my_ratings[1150] = 5   # Incredibles, The (2004)
        self._my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
        self._my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
        self._my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
        self._my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
        self._my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
        self._my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
        self._my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
        self._my_rated = [i for i in range(len(self._my_ratings)) if self._my_ratings[i] > 0]

        print('\nNew user ratings:\n')
        for i in range(len(self._my_ratings)):
            if self._my_ratings[i] > 0 :
                print(f'Rated {self._my_ratings[i]} for  {self._movieList_df.loc[i,"title"]}');

        # Reload ratings
        self._Y, self._R = self._load_ratings_small()

        # Add new user ratings to Y 
        self._Y = numpy.c_[self._my_ratings, self._Y]

        # Add new user indicator matrix to R
        self._R = numpy.c_[(self._my_ratings != 0).astype(int), self._R]

        # Normalize the Dataset
        self._normalizeRatings()

        #  Useful Values
        self._num_movies, self._num_users = self._Y.shape

        # Set Initial Parameters (W, X), use tf.Variable to track these variables
        tf.random.set_seed(1234) # for consistent results
        self._W = tf.Variable(tf.random.normal((self._num_users,  self._num_features),dtype=tf.float64),  name='W')
        self._X = tf.Variable(tf.random.normal((self._num_movies, self._num_features),dtype=tf.float64),  name='X')
        self._b = tf.Variable(tf.random.normal((1, self._num_users),   dtype=tf.float64),  name='b')
        print(f"W: {self._W.shape}, X: {self._X.shape}, b: {self._b.shape}, R: {self._R.shape}, Ynorm: {self._Ynorm.shape}")
        # Instantiate an optimizer.
        self._optimizer = keras.optimizers.Adam(learning_rate=1e-1)

    def _cofi_cost_func_v(self):
        """
        Returns the cost, J, for the content-based filtering
        Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
        Args:
        X (ndarray (num_movies,num_features)): matrix of item features
        W (ndarray (num_users,num_features)) : matrix of user parameters
        b (ndarray (1, num_users)            : vector of user parameters
        Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
        R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
        lambda_ (float): regularization parameter
        Returns:
        J (float) : Cost
        """
        j = (tf.linalg.matmul(self._X, tf.transpose(self._W)) + self._b - self._Ynorm) * self._R
        J = 0.5 * tf.reduce_sum(j**2) + (self._lambda/2) * (tf.reduce_sum(self._X**2) + tf.reduce_sum(self._W**2))
        return J
    
    def BuildModel(self):
        """
        Train the collaborative filtering model. This will learn the parameters  W, X and b.
        The operations involved in learning W, X and b simultaneously do not fall into the typical 'layers' offered in the TensorFlow neural network package. Consequently, the flow used in Course 2: Model, Compile(), Fit(), Predict(), are not directly applicable. Instead, we can use a custom training loop.

        Recall from earlier labs the steps of gradient descent.

        repeat until convergence:
        compute forward pass
        compute the derivatives of the loss relative to parameters
        update the parameters using the learning rate and the computed derivatives
        TensorFlow has the marvelous capability of calculating the derivatives for you. This is shown below. Within the tf.GradientTape() section, operations on Tensorflow Variables are tracked. When tape.gradient() is later called, it will return the gradient of the loss relative to the tracked variables. The gradients can then be applied to the parameters using an optimizer.
        Once a model is trained, the 'distance' between features vectors gives an indication of how similar items are. 
        """
        iterations = 200
        for iter in range(iterations):
            # Use TensorFlow’s GradientTape
            # to record the operations used to compute the cost 
            with tf.GradientTape() as tape:

                # Compute the cost (forward pass included in cost)
                cost_value = self._cofi_cost_func_v()

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss
            grads = tape.gradient( cost_value, [self._X, self._W, self._b] )

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self._optimizer.apply_gradients( zip(grads, [self._X, self._W, self._b]) )

            # Log periodically.
            if iter % 20 == 0:
                print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

    def Recommendations(self):
        """
        Compute the ratings for all the movies and users and display the movies that are recommended. These are based on the movies and ratings entered as self._my_ratings[] above. To predict the rating of movie i for user j, you compute numpy.dot(W[j,:], X[i,:]) + b[j]. This can be computed for all ratings using matrix multiplication.
        """
        # Make a prediction using trained weights and biases
        p = numpy.matmul(self._X.numpy(), numpy.transpose(self._W.numpy())) + self._b.numpy()

        #restore the mean
        pm = p + self._Ymean

        my_predictions = pm[:,0]

        # sort predictions
        ix = tf.argsort(my_predictions, direction='DESCENDING')

        for i in range(17):
            j = ix[i]
            if j not in self._my_rated:
                print(f'Predicting rating {my_predictions[j]:0.2f} for movie {self._movieList[j]}')

        print('\n\nOriginal vs Predicted ratings:\n')
        for i in range(len(self._my_ratings)):
            if self._my_ratings[i] > 0:
                print(f'Original {self._my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {self._movieList[i]}')
        """
        In practice, additional information can be utilized to enhance our predictions. Above, the predicted ratings for the first few hundred movies lie in a small range. We can augment the above by selecting from those top movies, movies that have high average ratings and movies with more than 20 ratings. This section uses a Pandas data frame which has many handy sorting features.
        """
        filter=(self._movieList_df["number of ratings"] > 20)
        self._movieList_df["pred"] = my_predictions
        self._movieList_df = self._movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
        self._movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)

if __name__ == "__main__":
    movie_recommendation = MovieRecommendation('./data/small_movie_list.csv', 100, 1)
    movie_recommendation.BuildModel()
    movie_recommendation.Recommendations()