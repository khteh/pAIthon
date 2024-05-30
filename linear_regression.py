import numpy as np
from io import StringIO

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
    c = np.linalg.lstsq(x_train, prices_train)[0]
     
    # fit a linear regression model to the data and get the coefficients

    # read in the test data and separate x_test from it
    x_test = np.genfromtxt(test_file, skip_header=1)
    prices_test = x_test[:, -1] # for last column which contains the price
    x_test = x_test[:, :-1] # for all but last column

    # print out the linear regression coefficients
    print(c)

    # this will print out the predicted prics for the two new cabins in the test data set
    print(x_test @ c)

find_best(X, y, c)
train_and_test()