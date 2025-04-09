import numpy, pandas
# https://github.com/NicolasHug/Surprise/issues/485#issuecomment-2787770326
from surprise import Dataset, Reader, KNNWithMeans, SVD
from surprise.model_selection import GridSearchCV
numpy._import_array()
"""
https://realpython.com/build-recommendation-engine-collaborative-filtering/
Mock up user-based similarity with one new user "E" who has rated only movie 1"
"""
def LoadData():
    return Dataset.load_builtin("ml-100k")

def ItemBasedSimilarityPrediction():
    """
    Memory Based
    The first category includes algorithms that are memory based, in which statistical techniques are applied to the entire 
    dataset to calculate the predictions.    
    """
    print(f"=== {ItemBasedSimilarityPrediction.__name__} ===")
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

def KNNWithMeansParametersTuning(data):
    """
    With a dict of all parameters, GridSearchCV tries all the combinations of parameters and reports the best parameters for any accuracy measure
    For example, you can check which similarity metric works best for your data in memory-based approaches:
    https://surprise.readthedocs.io/en/stable/model_selection.html
    """
    print(f"\n=== {KNNWithMeansParametersTuning.__name__} ===")
    # Load the builtin Movielens-100K data
    sim_options = {
        "name": ["msd", "cosine"],
        "min_support": [3, 4, 5],
        "user_based": [False, True],
    }
    param_grid = {"sim_options": sim_options}
    gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
    gs.fit(data)
    print(f"Score: {gs.best_score["rmse"]}, params: {gs.best_params["rmse"]}")

def SVDParametersTuning(data):
    """
    Model Based
    The second category covers the Model based approaches, which involve a step to reduce or compress the large but sparse user-item matrix.     
    
    Similarly, for model-based approaches, we can use Surprise to check which values for the following factors work best:
    - n_epochs is the number of iterations of SGD, which is basically an iterative method used in statistics to minimize a function.
    - lr_all is the learning rate for all parameters, which is a parameter that decides how much the parameters are adjusted in each iteration.
    - reg_all is the regularization term for all parameters, which is a penalty term added to prevent overfitting.
    """
    print(f"\n=== {SVDParametersTuning.__name__} ===")
    param_grid = {
        "n_epochs": [5, 10],
        "lr_all": [0.002, 0.005],
        "reg_all": [0.4, 0.6]
    }
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
    gs.fit(data)
    print(f"Score: {gs.best_score["rmse"]}, params: {gs.best_params["rmse"]}")

if __name__ == "__main__":
    data = LoadData()
    ItemBasedSimilarityPrediction()
    KNNWithMeansParametersTuning(data)
    SVDParametersTuning(data)