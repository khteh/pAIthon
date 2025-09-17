import numpy, pandas
import tensorflow as tf
# https://github.com/NicolasHug/Surprise/issues/485#issuecomment-2787770326
#from surprise import Dataset, Reader, KNNWithMeans, SVD
#from surprise.model_selection import GridSearchCV
#numpy._import_array()
"""
https://realpython.com/build-recommendation-engine-collaborative-filtering/
Mock up user-based similarity with one new user "E" who has rated only movie 1"

r[i,j] = 1 if user j has rated movie i, 0 otherwise
y[i,j] = rating given by user j for movie i, if r[i,j] = 1
w[j], b[j] = parameters for user j
x[i] = features of movie i
m[j] = # movies rated by user j
n = # features
Nu = # users
Nm = # movies
Cost function for a single user: J(w[j],b[j]) = sum((w[j] * x[i] + b[i] - y[i,j]) ** 2) / 2m[j] + lambda * sum(w[j] ** 2) / 2m[j] for all r[i,j] == 1
Cost function for all users: J(w,b) = sum(sum((w[j] * x[i] + b[i] - y[i,j])) ** 2) / 2m[j] + lambda * sum(sum(w[j] ** 2)) / 2m[j] for all r[i,j] == 1

Suppose we know the w[j], b[j], we can estimate features using these cost functions
Cost function for a single feature: J(x[j]) = sum((w[j] * x[i] + b[i] - y[i,j]) ** 2) / 2m[j] + lambda * sum(x[i] ** 2) / 2m[j] for all r[i,j] == 1
Cost function for all features: J(x) = sum(sum((w[j] * x[i] + b[i] - y[i,j])) ** 2) / 2m[j] + lambda * sum(sum(x[i] ** 2)) / 2m[j] for all r[i,j] == 1
This is only possible in Collaborative filtering because we have relevant data to do this "reverse engineering" from Y to features. Not possible in linear regression.

Putting the 2 above together, we have:
J(w,b,x) = sum((w[j] * x[i] + b[j] - y[i,j]) ** 2) / 2 + lambda * sum(sum(w[j] ** 2)) / 2 + lambda * sum(sum(x[i] ** 2)) / 2 for all r[i,j] == 1. Ignore /m[j] to simplify it.
With this, we could use Gradient Descent optimizer to learn w,b AND x. Now, x, the feature becomes a parameter. This is only possible in Collaborative Filtering because of the nature of input data.

This works also for binary classification by using g(z).

Mean Normalization: Makes training faster and prediction of new row / column more reasonable instead of '0'. Without the mean of the rows/cols, 'w' ~= 0 (because of regularization) and therefore, 'z' is likely to be 0.

To find item k with x[k] similar to x[i]: Min(sum((x[k] - x[i]) ** 2))

Limitations:
(1) Cold start problem. For example,
    (i) Rank new items that few users have rated.
    (ii) Show something reasonable to new users who have rated only a few items.
(2) It doesn't give a natural way to use side information or additional information about items or users. For example,
    (i) Items: Genre, movie stars, studio, authors, etc.
    (ii) Users: Demographics (age, gender, location), expressed preferences, types of devices/browsers used to access the application, etc.
"""
class CollaborativeFiltering():
    _data = None
    def __init__(self):
        self._prepare_data()

    def _prepare_data(self):
        self._data = Dataset.load_builtin("ml-100k")

    def cofi_cost_func(self, X, W, b, Y, R, lambda_):
        """
        Returns the cost for the content-based filtering
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
        nm, nu = Y.shape
        J = 0
        param = 0
        w_sum = numpy.sum(numpy.square(W))
        x_sum = numpy.sum(numpy.square(X))
        # 𝐗: nm x 10; W: nu x 10
        for j in range(nu):
            w = W[j,:]
            b_j = b[0,j]
            for i in range(nm):
                if R[i,j]:
                    x = X[i,:]
                    y = Y[i,j]
                    r = R[i,j]
                    param += ((w @ x) + b_j - y) ** 2
        print(f"param: {param}, w_sum: {w_sum}, x_sum: {x_sum}")
        J = (param + w_sum * lambda_ + x_sum * lambda_) / 2
        print(f"J: {J}")
        return J

    def ItemBasedSimilarityPrediction(self):
        """
        Memory Based
        The first category includes algorithms that are memory based, in which statistical techniques are applied to the entire 
        dataset to calculate the predictions.    
        """
        print(f"=== {self.ItemBasedSimilarityPrediction.__name__} ===")
        ratings_dict = {
            "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
            "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
            "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],
        }
        df = pandas.DataFrame(ratings_dict)
        reader = Reader(rating_scale=(1,5))
        # Load Pandas dataframe
        similar_ratings = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

        sim_options = {
            "name": "cosine",
            "user_based": False, # Compute similarities between items
        }
        algo = KNNWithMeans(sim_options=sim_options)
        train = similar_ratings.build_full_trainset()
        algo.fit(train)
        predict = algo.predict('E', 2) # algo.predict(uid, iid) -- predict the rating score of the iid of uid
        print(f"User E is predicted to rate movie 2 as {predict.est}")

    def KNNWithMeansParametersTuning(self):
        """
        With a dict of all parameters, GridSearchCV tries all the combinations of parameters and reports the best parameters for any accuracy measure
        For example, you can check which similarity metric works best for your data in memory-based approaches:
        https://surprise.readthedocs.io/en/stable/model_selection.html
        """
        print(f"\n=== {self.KNNWithMeansParametersTuning.__name__} ===")
        # Load the builtin Movielens-100K data
        sim_options = {
            "name": ["msd", "cosine"],
            "min_support": [3, 4, 5],
            "user_based": [False, True],
        }
        param_grid = {"sim_options": sim_options}
        gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
        gs.fit(self._data)
        print(f"Score: {gs.best_score["rmse"]}, params: {gs.best_params["rmse"]}")

    def SVDParametersTuning(self):
        """
        Model Based
        The second category covers the Model based approaches, which involve a step to reduce or compress the large but sparse user-item matrix.     
        
        Similarly, for model-based approaches, we can use Surprise to check which values for the following factors work best:
        - n_epochs is the number of iterations of SGD, which is basically an iterative method used in statistics to minimize a function.
        - lr_all is the learning rate for all parameters, which is a parameter that decides how much the parameters are adjusted in each iteration.
        - reg_all is the regularization term for all parameters, which is a penalty term added to prevent overfitting.
        """
        print(f"\n=== {self.SVDParametersTuning.__name__} ===")
        param_grid = {
            "n_epochs": [5, 10],
            "lr_all": [0.002, 0.005],
            "reg_all": [0.4, 0.6]
        }
        gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
        gs.fit(self._data)
        print(f"Score: {gs.best_score["rmse"]}, params: {gs.best_params["rmse"]}")

if __name__ == "__main__":
    collaborativeFiltering = CollaborativeFiltering()
    collaborativeFiltering.ItemBasedSimilarityPrediction()
    collaborativeFiltering.KNNWithMeansParametersTuning()
    collaborativeFiltering.SVDParametersTuning()