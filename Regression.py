import numpy, math, copy
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from io import StringIO
from CostIterationPlot import CostIterationPlot
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

https://realpython.com/linear-regression-in-python/
To get the best weights, you usually minimize the sum of squared residuals (SSR) for all observations 𝑖 = 1, …, 𝑛: SSR = Σᵢ(𝑦ᵢ - 𝑓(𝐱ᵢ))². This approach is called the method of ordinary least squares.

Regression Performance
The variation of actual responses 𝑦ᵢ, 𝑖 = 1, …, 𝑛, occurs partly due to the dependence on the predictors 𝐱ᵢ. However, there’s also an additional inherent variance of the output.
The coefficient of determination, denoted as 𝑅², tells you which amount of variation in 𝑦 can be explained by the dependence on 𝐱, using the particular regression model. A larger 𝑅² indicates a better fit and means that the model can better explain the variation of the output with different inputs.
The value 𝑅² = 1 corresponds to SSR = 0. That’s the perfect fit, since the values of predicted and actual responses fit completely to each other.
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
X = numpy.array([[66, 5, 15, 2, 500], 
              [21, 3, 50, 1, 100], 
              [120, 15, 5, 2, 1200]])
y = numpy.array([250000, 60000, 525000])

# alternative sets of coefficient values
c = numpy.array([[3000, 200 , -50, 5000, 100], 
              [2000, -250, -100, 150, 250], 
              [3000, -100, -150, 0, 150]])   

def find_best(X, y, c):
    smallest_error = numpy.inf
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
    numpy.set_printoptions(precision=1)    # this just changes the output settings for easier reading
    train_file = StringIO(train_string) # simulate reading a file
    test_file = StringIO(test_string) # simulate reading a file
    # read in the training data and separate it to x_train and y_train
    x_train = numpy.genfromtxt(train_file, skip_header=1)
    prices_train = x_train[:, -1] # for last column which contains the price
    x_train = x_train[:, :-1] # for all but last column
    #print(f"x: {x}, prices: {prices}")
    c = numpy.linalg.lstsq(x_train, prices_train, rcond=None)[0]
     
    # fit a linear regression model to the data and get the coefficients

    # read in the test data and separate x_test from it
    x_test = numpy.genfromtxt(test_file, skip_header=1)
    # Slicing 2-D arrays:
    prices_test = x_test[:, -1] # For all the subarrays, get the last column which contains the price
    x_test = x_test[:, :-1] # For all the subarrays, get all the elements except the last column

    # print out the linear regression coefficients
    print(c)

    # this will print out the predicted prics for the two new cabins in the test data set
    print(x_test @ c) # Matrix multiplication

def ZScoreNormalization(x):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature

    Z-score normalization: Find the standard deviation (d) and the mean. (x - ave) / d.
    Mean: sum(data) / len(data)
    standard deviation: sum((data[i] - mean) ** 2) / len(data)
    """
    # Find the mean of each feature (column)
    # The axis=0 argument specifies that the accumulation should occur along the rows, effectively accumulating values down each column.
    mu = numpy.mean(x, axis=0) # mu will have shape (n,)
    # Find the standard deviation of each feature (column)
    sigma = numpy.std(X, axis=0) # sigma will have shape (n,)
    # Subtract every element of mu, divide by sigma
    x_normalized = (x - mu)  / sigma
    return x_normalized, mu, sigma

def SquaredErrorCostFunction(x, y, w: float, b:float):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters
     lambda_ (scalar): Controls amount of regularization
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y

    sum((predictons[i] - targets[i]) ** 2) / 2n
    n: number of observations
    2: Further division by 2 is just to make the error number neat without affecting the modal performance measurement.
    J(w,b) = (sum((f_w_b(x) - y) ** 2)) / 2m (MeanSquaredError)
    Squared error cost will never have multiple local minimums. Only ONE single global minimum. In 3D plot, it is a bowl shape. It is a convex function.
    """
    # number of training examples
    m = x.shape[0]
    cost = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost += (f_wb - y[i]) ** 2
    cost /= (2 * m)
    return cost

def MultipleLinearRegressionSquaredErrorCostFunction(x, y, w, b: float, lambda_: float = 1.0):
    """
    Computes the cost function for multiple linear regression.
    
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y

    sum((predictons[i] - targets[i]) ** 2) / 2n
    n: number of observations
    2: Further division by 2 is just to make the error number neat without affecting the modal performance measurement.
    J(w,b) = (sum((f_w_b(x) - y) ** 2)) / 2m (MeanSquaredError)
    Regularized: J(w,b) = (sum((f_w_b(x) - y) ** 2) + lambda * sum(w ** 2)) / 2m
    Squared error cost will never have multiple local minimums. Only ONE single global minimum. In 3D plot, it is a bowl shape. It is a convex function.
    """
    # number of training examples
    m = x.shape[0] 
    cost = 0
    rcost = 0
    for i in range(m): 
        f_wb_i = numpy.dot(x[i], w) + b   
        cost += (f_wb_i - y[i]) ** 2
    for i in w:
        rcost += i ** 2
    rcost *= lambda_
    cost += rcost
    cost /= (2 * m)
    return cost

def UniVariateLinearRegressionGradient(x, y, w: float, b: float): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    
    dJ(w,b)/dw = (sum((f_w_b(x) - y) * x)) / m
    dJ(w,b)/db = (sum((f_w_b(x) - y))) / m
    Gradient descent is picking the 'correct' features for us by emphasizing its associated parameter.
    Less weight value implies less important/correct feature, and in extreme, when the weight becomes zero or very close to zero, the associated feature is not useful in fitting the model to the data.
    In Tensorflow, derivatives are calculated using back-propagation
    """
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    for i in range(m):  
        f_wb = w * x[i] + b
        err = f_wb - y[i]
        dj_dw += err * x[i] 
        dj_db += err 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
    return dj_dw, dj_db

def MultipleLinearRegressionGradient(X, y, w, b: float, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.

    dJ(w,b)/dw = (sum((f_w_b(x) - y) * x) + lambda * w) / m
    dJ(w,b)/db = (sum((f_w_b(x) - y))) / m
    Gradient descent is picking the 'correct' features for us by emphasizing its associated parameter.
    Less weight value implies less important/correct feature, and in extreme, when the weight becomes zero or very close to zero, the associated feature is not useful in fitting the model to the data.
    In Tensorflow, derivatives are calculated using back-propagation
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = numpy.zeros((n,))
    dj_db = 0.
    for i in range(m):                             
        err = (numpy.dot(X[i], w) + b) - y[i]
        for j in range(n):                         
            dj_dw[j] += err * X[i, j] + lambda_ * w[j]
        dj_db += err                        
    dj_dw /= m                                
    dj_db /= m                                
    return dj_db, dj_dw

def UniVariateLinearRegressionGradientDescent(x, y, w_in: float, b_in: float, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)

        # Update Parameters using equation (3) above
        b -= alpha * dj_db
        w -= alpha * dj_dw

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history #return w and J,w history for graphing

def MultipleLinearRegressionGradientDescent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w -= alpha * dj_dw               ##None
        b -= alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history #return final w,b and J history for graphing

def UniVariateLinearRegressionTraining():
    """
    UniVariate Linear regression training using local implementation of Cost and gradient descent.
    """
    # Load our data set
    x_train = numpy.array([1.0, 2.0])   #features
    y_train =  numpy.array([300.0, 500.0])   #target value    
    # initialize parameters
    w_init = 0
    b_init = 0
    # some gradient descent settings
    iterations = 10000
    """
    One way to debug if gradient descent works is setting learning rate to a very small value and check if cost decreases after every iteration.
    Example: start from 0.001, 0.003 (x3), 0.01 (x3), 0.03 (x3), 0.1 (x3), 1.0 (x3) ...
    """
    alpha = 1.0e-2
    # run gradient descent
    w_final, b_final, J_hist, p_hist = UniVariateLinearRegressionGradientDescent(x_train ,y_train, w_init, b_init, alpha, 
                                                        iterations, SquaredErrorCostFunction, UniVariateLinearRegressionGradient)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
    CostIterationPlot(J_hist)

def MultipleLinearRegressionTraining():
    """
    Multiple Linear regression training using local implementation of Cost and gradient descent.
    """
    X_train = numpy.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = numpy.array([460, 232, 178])
    # initialize parameters
    b_init = 785.1811367994083
    w_init = numpy.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    initial_w = numpy.zeros_like(w_init) # Return an array of zeros with the same shape and type as a given array.
    initial_b = 0.
    # some gradient descent settings
    iterations = 1000
    """
    One way to debug if gradient descent works is setting learning rate to a very small value and check if cost decreases after every iteration.
    Example: start from 0.001, 0.003 (x3), 0.01 (x3), 0.03 (x3), 0.1 (x3), 1.0 (x3) ...
    """
    alpha = 5.0e-7
    # run gradient descent 
    w_final, b_final, J_hist = MultipleLinearRegressionGradientDescent(X_train, y_train, initial_w, initial_b,
                                                        MultipleLinearRegressionSquaredErrorCostFunction, MultipleLinearRegressionGradient, 
                                                        alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {numpy.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    CostIterationPlot(J_hist)

def SimpleLinearRegression():
    """
    Only 1 input
    f(x) = b0 + b1x0
    """
    print(f"\n=== {SimpleLinearRegression.__name__} ===")
    """
    .reshape() on x because this array must be two-dimensional, or more precisely, it must have one column and as many rows as necessary.
    One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
    """
    x = numpy.array([5,15,25,35,45,55]).reshape((-1, 1))
    y = numpy.array([5,20,14,32,22,38])
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    model = LinearRegression(n_jobs=-1) # Use all available processors
    model.fit(x, y)
    print(f"b0: {model.intercept_}, b1: {model.coef_}")
    r_squared = model.score(x, y) # 'r' = residual. Sum of squared residuals
    print(f"r_squared: {r_squared}") # 1 means perfect fit/overfitting. Low value = underfitting => Linear regression model is not right for the problem in this case.
    predictions = model.predict(x)
    print(f"Predictions: {predictions}")
    # Do it manually according to the mathematical model of simple linear regression:
    manual_predictions = model.intercept_ + model.coef_ * x
    print(f"Manual Predictions: {manual_predictions}")
    x = numpy.arange(5).reshape((-1,1))
    y = model.predict(x)
    manual_predictions = model.intercept_ + model.coef_ * x
    print(f"Predictions of new input: {y}, {manual_predictions}")

def MultipleLinearRegression():
    """
    Linear regression with > 1 feature.
    Input is at least 2-dimensional
    f(x) = b0 + b1x0 + ... + bnxn
    n obvservations: (x1,y1), (x2,y2),(x3,y3),...,(xn,yn)
    Each observation for the input variable is an n-dimensional array: xi = (xi,1 xi,2 xi,3 ... xi,n). i = ith observation, the # after i is the component of the input.
    """
    """
    8 observations
    Each input observation consists of 2 data points => 2D linear regression model.
    """
    print(f"\n=== {MultipleLinearRegression.__name__} ===")
    x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
    y = [4, 5, 20, 14, 32, 22, 38, 43]
    x = numpy.array(x)
    y = numpy.array(y)
    print(f"x: {x.shape}, y: {y.shape}")
    model = LinearRegression(n_jobs=-1).fit(x,y)
    r_squared = model.score(x,y) # 'r' = residual. Sum of squared residuals
    print(f"r_squared: {r_squared}") # 1 means perfect fit/overfitting. Low value = underfitting => Linear regression model is not right for the problem in this case.
    print(f"b0: {model.intercept_}, coefficients: {model.coef_}")
    predictions = model.predict(x)
    residuals = y - predictions
    print(f"y:           {y}")
    print(f"predictions: {predictions}")
    print(f"residuals:   {residuals}")
    x = numpy.arange(10).reshape((-1, 2)) # One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
    predictions = model.predict(x)
    print(f"predictions: {predictions}")

def SimplePolynomialRegression():
    """
    f(x) = b0 + b1x + b2x^2 + ... + bnx^n
    With only 1 independent variable x, we seek a regression model of the form f(x) = b0 + b1x + b2x^2 + ... + bnx^n
    Involves 1 extra step of calculating the higher degrees of the input variable value.
    Preprocessing of the input observations in order to satisfy the polynomial equation, i.e., to derive the high-degree values
    For example, quadratic model will have 2 features of the input, Qubic is 3, etc.

    Polynomial features were chosen based on how well they matched the target data. Another way to think about this is to note that we are still using linear regression once we have created new features. Given that, the best features will be linear relative to the target.
    """
    print(f"\n=== {SimplePolynomialRegression.__name__} ===")
    x = numpy.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1)) # 1 input with 6 observations
    y = numpy.array([15, 11, 2, 8, 25, 32])
    transformer = PolynomialFeatures(degree = 2, include_bias=False) # Quadratic. include_bias=False generates X without the column with all '1's
    print("x:")
    print(x)
    x_transformed = transformer.fit_transform(x)
    print("\nx_transformed:")
    print(x_transformed)
    model = LinearRegression(n_jobs=-1).fit(x_transformed, y)
    r_squared = model.score(x_transformed,y) # 'r' = residual. Sum of squared residuals
    print(f"r_squared: {r_squared}") # 1 means perfect fit/overfitting. Low value = underfitting => Linear regression model is not right for the problem in this case.
    print(f"b0: {model.intercept_}, coefficients: {model.coef_}")
    predictions = model.predict(x_transformed)
    residuals = y - predictions
    print(f"y:           {y}")
    print(f"predictions: {predictions}")
    print(f"residuals:   {residuals}")

def MultiplePolynomialRegression():
    """
    f(x) = b0 + b1x1 + b2x2 + b3x1^2 + b4x1x2 + b5x2^2
    Polynomial features were chosen based on how well they matched the target data. Another way to think about this is to note that we are still using linear regression once we have created new features. Given that, the best features will be linear relative to the target.
    """
    print(f"\n=== {MultiplePolynomialRegression.__name__} ===")
    x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
    y = [4, 5, 20, 14, 32, 22, 38, 43]
    x = numpy.array(x)
    y = numpy.array(y)
    transformer = PolynomialFeatures(degree = 2, include_bias=False) # Quadratic. include_bias=False generates X without the column with all '1's
    x_transformed = transformer.fit_transform(x)
    print("\nx_transformed:")
    print(x_transformed)
    model = LinearRegression(n_jobs=-1).fit(x_transformed, y)
    r_squared = model.score(x_transformed,y) # 'r' = residual. Sum of squared residuals
    print(f"r_squared: {r_squared}") # 1 means perfect fit/overfitting. Low value = underfitting => Linear regression model is not right for the problem in this case.
    print(f"b0: {model.intercept_}, coefficients: {model.coef_}")
    predictions = model.predict(x_transformed)
    residuals = y - predictions
    print(f"y:           {y}")
    print(f"predictions: {predictions}")
    print(f"residuals:   {residuals}")
    x_new = numpy.range(10).reshape((-1, 2))
    print("x_new:")
    print(x_new)
    predictions = model.predict(x_new)
    print(f"predictions: {predictions}")

def MultiplePolynomialRegressionStatsModels():
    """
    f(x) = b0 + b1x1 + b2x2 + b3x1^2 + b4x1x2 + b5x2^2
    statsmodels provides advanced statistical parameters of a model prediction result. Otherwise it is almost the same as scikit-learn.
    """
    print(f"\n=== {MultiplePolynomialRegressionStatsModels.__name__} ===")
    x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
    y = [4, 5, 20, 14, 32, 22, 38, 43]
    x = numpy.array(x)
    y = numpy.array(y)
    """
    Add the column of ones to the inputs if you want statsmodels to calculate the intercept 𝑏₀. It doesn’t take 𝑏₀ into account by default.
    returns a new array with the column of ones inserted at the beginning.
    """
    x = sm.add_constant(x)
    print("x:")
    print(x)
    model = sm.OLS(y, x)   
    result: statsmodels.regression.linear_model.RegressionResultsWrapper = model.fit()
    print("results: ")
    print(result.summary())
    predictions = model.predict(x)
    residuals = y - predictions
    print(f"y:           {y}")
    print(f"predictions: {predictions}")
    print(f"residuals:   {residuals}")
    x_new = numpy.range(10).reshape((-1, 2))
    print("x_new:")
    print(x_new)
    predictions = model.predict(x_new)
    print(f"predictions: {predictions}")

if __name__ == "__main__":
    find_best(X, y, c)
    train_and_test()
    UniVariateLinearRegressionTraining()
    MultipleLinearRegressionTraining()
    SimpleLinearRegression()
    MultipleLinearRegression()
    SimplePolynomialRegression()
    MultiplePolynomialRegression()
    MultiplePolynomialRegressionStatsModels()   