import numpy, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from utils.FileUtil import Download
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
rng = numpy.random.default_rng(seed=19)
"""
https://realpython.com/videos/knn-python-overview/
Unlike other supervised learning algos, kNN perform calculations during prediction.
Can be used for both classification and regression, non-linear and non-parametric.
Non-linear:
(i) Classification: Decision boundaries need not follow a line/hyperplane
(ii) Regression: Predictions do not need to be linear or even monotonic with respect to input data.

Non-parametric:
(i) No model parameters to fit during trainig
(ii) Predictions made directly from training data
(iii) Highly flexible and logic is intuitive
(iv) Must keep training dataset
    (a) Memory intensive
    (b) Predictions may be slow for large training datasets

Adapts as new training observations are collected. Good for continuous flow of incoming data.
Essentially no training time needed. It only needs to store the training datasets.

Classification:
- Use majority vote: Majority of the k-nearest neighbours' Label values.

Regression:
- Use average of the k-nearest neighbours' Label values.

Hyperparameter K should be tuned for each new problem.
k = 1: High variance
k = N: High bias
To find optimized k, use validation set or a cross-validation method like .GridSearchCV()

Drawbacks:
(1) Lazy learner: All work is done at prediction time
(2) Slow: Prediction time scales linearly with training set size.
(3) Memory intensive: training datasets must be retained for prediction. Not feasible for edge computing or small devices?
(4) Features should be scaled to judge distances fairly. If some data are small values and others are big, the big data points will bias / dominate the distance calculations.
"""
def EuclideanDistance(a, b):
    """
    dist(A,B) = sqrt((a1 - b1)**2 + ... + (an - bn)**2)
    Norm of a - b => vector distance
    """
    sum = 0
    for ai, bi in zip(a, b):
        sum = sum + (ai - bi)**2
    return numpy.sqrt(sum)

def Nearest():
    print(f"=== {Nearest.__name__} ===")
    x_train = rng.random((10, 3))   # generate 10 random vectors of dimension 3
    x_test = rng.random(3)        # generate one more random vector of the same dimension
    print("x_train:")
    print(x_train)
    print("x_test: ")
    print(x_test)
    nearest_index = -1
    min_distance = numpy.Inf
    # add a loop here that goes through all the vectors in x_train and finds the one that
    # is nearest to x_test. return the index (between 0, ..., len(x_train)-1) of the nearest
    # neighbor
    for i in range(0, len(x_train)):
        distance = EuclideanDistance(x_train[i], x_test)
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
    print(f"Index of the nearest neighbour: {nearest_index}")

def kNNClassificationNumpy():
    print(f"\n=== {kNNClassificationNumpy.__name__} ===")
    # To do classification:
    nearest_neighbour_classes = numpy.array(["Square", "Circle", "Square", "Triangle", "Square", "Circle"])
    class_count = Counter(nearest_neighbour_classes)
    print(f"\nMost common class: {class_count.most_common(1)}")

def AbaloneAgeNumpy(X, Y, toPredict, k) -> float:
    # Calculate distances between toPredict and every other data points in the dataset
    distances = numpy.linalg.norm(X - toPredict[0], axis=1)
    #print(f"\nDistances (shape: {distances.shape}):")
    print(distances)
    # Find nearest neighbours. Use numpy.argsort which returns the indices that would sort the array.
    distances_sorted_indices = distances.argsort()[:k]
    """
    print(f"\n{k}-closest neighbours:")
    for i in distances_sorted_indices:
        print(f"i: {i}")
        print(f"X: {X[i]}")
        print(f"distance: {distances[i]}")
    """
    # Regression: Average neighbours' labels
    nearest_neighbour_rings = Y[distances_sorted_indices]
    print("\nNearest neighbours' rings:")
    print(nearest_neighbour_rings)
    return nearest_neighbour_rings.mean()

def AbaloneAgeScikitLearn(X, Y, toPredict, k) -> float:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=135790) # random_state will fix the split result sets. Otherwise, every run will produce different split result sets.
    #print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, toPredict: {toPredict.shape}")
    kNN_model = KNeighborsRegressor(n_neighbors=k)
    kNN_model.fit(X_train, Y_train)
    train_predictions = kNN_model.predict(X_train)
    test_predictions = kNN_model.predict(X_test)
    # Score the predictions
    train_mean_squared_errors = mean_squared_error(Y_train, train_predictions)
    train_errors_root_mean_squared = numpy.sqrt(train_mean_squared_errors)
    test_mean_squared_errors = mean_squared_error(Y_test, test_predictions)
    test_errors_root_mean_squared = numpy.sqrt(test_mean_squared_errors)
    print(f"\nRoot-mean-squared errors of train: {train_errors_root_mean_squared}, test: {test_errors_root_mean_squared}")
    return kNN_model.predict(toPredict)

def AbaloneAge(url, path):
    """
    An abalone's rings indicate its age but is difficult for biologists to obtain
    Goal: Predict age (rings) from abalone's physical measurement using numpy and scikit-learn
    """
    print(f"\n=== {AbaloneAge.__name__} ===")
    Download(url, Path(path))
    data = pd.read_csv(path, header=None)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data.columns = ["Sex", "Length", "Diameter", "Height", "Whole Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "Rings"]
    print(data.head())
    # Predict based on physical measurement. Therefore, drop the "Sex" column
    data.drop(labels=['Sex'], axis=1, inplace=True)
    print(data.head())
    print("\ndata.info():")
    data.info()
    # Label to predict: "Rings"
    print("\nLabel(Rings) info:")
    print(data["Rings"].describe())
    plt.hist(data["Rings"], bins=15) # bins is by trial and error
    plt.xlabel("Count")
    plt.ylabel("Rings")
    plt.title("Abalone ring counts distribution")
    #plt.show()# This blocks
    # Correlations between input variables and target output (Label)
    correlations = data.corr()
    print("\nCorrelations between input variables:")
    print(correlations)
    print("\nCorrelations between input variables and target output (Label):")
    print(correlations["Rings"])
    # If there are a few variables which are highly correlated to the Label, then the ML would be straight-forward and can use linear regression instead.
    X = data.drop("Rings", axis=1) # Features
    X_array = X.values
    print(f"\nFeatures (shape: {X_array.shape}):")
    print(X_array)
    Y = data["Rings"] # Label
    Y_array = Y.values
    print(f"\nLabel (shape: {Y_array.shape}):")
    print(Y_array)
    print(f"feature count: {X_array.shape[1]}")
    toPredict = rng.random((1, X_array.shape[1]))
    print(f"\nTo Predict (shape: {toPredict.shape}):")
    print(toPredict)
    print(f"Rings prediction using numpy (k=3): {AbaloneAgeNumpy(X_array, Y_array, toPredict, 3)}")
    print(f"Rings prediction using numpy (k=10): {AbaloneAgeNumpy(X_array, Y_array, toPredict, 10)}")
    print(f"Rings prediction using scikit-learn (k=3): {AbaloneAgeScikitLearn(X_array, Y_array, toPredict, 3)}")
    print(f"Rings prediction using scikit-learn (k=10): {AbaloneAgeScikitLearn(X_array, Y_array, toPredict, 10)}")

if __name__ == "__main__":
    Nearest()
    kNNClassificationNumpy()
    AbaloneAge("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "/tmp/abalone.data")
