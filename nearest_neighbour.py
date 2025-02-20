import numpy, seaborn, pandas as pd
from pathlib import Path
from downloads import download_file
rng = numpy.random.default_rng(seed=19)
def dist(a, b):
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
        distance = dist(x_train[i], x_test)
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
    print(f"Index of the nearest neighbour: {nearest_index}")

def AbaloneAge(url, path):
    """
    Goal: Predict age (rings) from abalone's physical measurement
    """
    print(f"\n=== {AbaloneAge.__name__} ===")
    download_file(url, Path(path))
    data = pd.read_csv(path, header=None)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data.columns = ["Sex", "Length", "Diameter", "Height", "Whole Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "Rings"]
    print(data.head())
    # Predict based on physical measurement. Therefore, drop the "Sex" column
    data.drop(labels=['Sex'], axis=1, inplace=True)
    print(data.head())
    print("\ndata.info():")
    print(data.info())
    # Label to predict: "Rings"
    print("\nLabel(Rings) info:")
    print(data["Rings"].describe())
    seaborn.histplot(data["Rings"])
    # Correlations between input variables and target output (Label)
    correlations = data.corr()
    print("\nCorrelations between input variables:")
    print(correlations)
    print("\nCorrelations between input variables and target output (Label):")
    print(correlations["Rings"])
    # If there are a few variables which are highly correlated to the Label, then the ML would be straight-forward and can use linear regression instead.

if __name__ == "__main__":
    Nearest()
    AbaloneAge("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "/tmp/abalone.data")
