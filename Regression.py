import numpy as np
from sklearn.linear_model import LinearRegression
from io import StringIO

"""
Goal: Build a mathematical model describing the effect of a set of input variables x1,x2,...,xn on another variable y.
x1,x2,...,xn are called predictors, independent variables, features.
y = f(x) + e
So, the main goal is to build a good model for f(x)
Observations: (x1,y1),(x2,y2),...,(xn,yn). The gist of a regression technique is in taking the n observations to build the model f(n).

Types of regression: Linear, polynomial, nonlinear, decision tree, support vector machines, NN.

2 main use cases of regression are:
(1) Prediction: Forecast the outcome of an event/state/object from previous knowledge. This use case has substantial overlap with ML.
(2) Inference: Determine how an event/state/object affects the production of another event/state/object
"""
train_string = '''
25 2 50 1 500 127900
39 3 10 1 1000 222100
13 2 13 1 1000 143750
82 5 20 2 120 268000
130 6 10 2 600 460700
115 6 10 1 550 407000
'''

test_string = '''
36 3 15 1 850 196000
75 5 18 2 540 290000
'''

# data
X = np.array([[66, 5, 15, 2, 500], 
              [21, 3, 50, 1, 100], 
              [120, 15, 5, 2, 1200]])
y = np.array([250000, 60000, 525000])

# alternative sets of coefficient values
c = np.array([[3000, 200 , -50, 5000, 100], 
              [2000, -250, -100, 150, 250], 
              [3000, -100, -150, 0, 150]])   

def find_best(X, y, c):
    smallest_error = np.Inf
    best_index = -1
    for i in range(0, len(c)):
        print(f"index {i}")
        dotproduct = c[i] @ X[i]
        print(f"dotproduct: {dotproduct}")
        squared = (dotproduct - y[i]) ** 2
        if squared < smallest_error:
            smallest_error = squared
            best_index = i
             # edit here: calculate the sum of squared error with coefficient set coeff and
                 # keep track of the one yielding the smallest squared error
    print("the best set is set %d" % best_index)

def train_and_test():
    np.set_printoptions(precision=1)    # this just changes the output settings for easier reading
    train_file = StringIO(train_string) # simulate reading a file
    test_file = StringIO(test_string) # simulate reading a file
    # read in the training data and separate it to x_train and y_train
    x_train = np.genfromtxt(train_file, skip_header=1)
    prices_train = x_train[:, -1] # for last column which contains the price
    x_train = x_train[:, :-1] # for all but last column
    #print(f"x: {x}, prices: {prices}")
    c = np.linalg.lstsq(x_train, prices_train, rcond=None)[0]
     
    # fit a linear regression model to the data and get the coefficients

    # read in the test data and separate x_test from it
    x_test = np.genfromtxt(test_file, skip_header=1)
    # Slicing 2-D arrays:
    prices_test = x_test[:, -1] # For all the subarrays, get the last column which contains the price
    x_test = x_test[:, :-1] # For all the subarrays, get all the elements except the last column

    # print out the linear regression coefficients
    print(c)

    # this will print out the predicted prics for the two new cabins in the test data set
    print(x_test @ c)

def SimpleLinearRegression():
    """
    Only 1 input
    f(x) = b0 + b1x0
    """
    print(f"\n=== {SimpleLinearRegression.__name__} ===")
    x = np.array([5,15,25,35,45,55]).reshape((-1, 1)) # One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
    y = np.array([5,20,14,32,22,38])
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    model = LinearRegression()
    model.fit(x, y)
    print(f"b0: {model.intercept_}, b1: {model.coef_}")
    r_squared = model.score(x, y)
    print(f"r_squared: {r_squared}") # 1 means perfect fit
    predictions = model.predict(x)
    print(f"Predictions: {predictions}")
    # Do it manually according to the mathematical model of simple linear regression:
    manual_predictions = model.intercept_ + model.coef_ * x
    print(f"Manual Predictions: {manual_predictions}")
    x = np.arange(5).reshape((-1,1))
    y = model.predict(x)
    manual_predictions = model.intercept_ + model.coef_ * x
    print(f"Predictions of new input: {y}, {manual_predictions}")

def MultipleLinearRegression():
    """
    Input is at least 2-dimensional
    f(x) = b0 + b1x0 + ... + bnxn
    n obvservations: (x1,y1), (x2,y2),(x3,y3),...,(xn,yn)
    Each observation for the input variable is an n-dimensional array: xi = (xi,1 xi,2 xi,3 ... xi,n). i = ith observation
    """
    """
    8 observations
    Each input observation consists of 2 data points => 2D linear regression model.
    """
    print(f"\n=== {MultipleLinearRegression.__name__} ===")
    x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
    y = [4, 5, 20, 14, 32, 22, 38, 43]
    x = np.array(x)
    y = np.array(y)
    print(f"x: {x.shape}, y: {y.shape}")
    model = LinearRegression().fit(x,y)
    r_squared = model.score(x,y)
    print(f"r_squared: {r_squared}") # 1 means perfect fit
    print(f"b0: {model.intercept_}, coefficients: {model.coef_}")
    predictions = model.predict(x)
    print(f"y:           {y}")
    print(f"predictions: {predictions}")
    x = np.arange(10).reshape((-1, 2)) # One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
    predictions = model.predict(x)
    print(f"predictions: {predictions}")

if __name__ == "__main__":
    find_best(X, y, c)
    train_and_test()
    SimpleLinearRegression()
    MultipleLinearRegression()