import numpy as np
import matplotlib.pyplot as plt
from utils import *
from F1Score import F1Score

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n = X.shape
    #print(f"m: {m}, n: {n}")
    ### START CODE HERE ### 
    sum = np.sum(X, axis=0) # axis=0
    #print(f"sum: {sum[:10]}, m: {m}")
    miu = sum / m
    #print(f"sum: {sum[:10]}, miu: {miu[:10]}")
    variances = np.zeros(n)
    #print(f"variances: {variances.shape}")
    for feature in range(n):
        for row in range(m):
            variances[feature] += (X[row][feature] - miu[feature]) ** 2
        variances[feature] /= m
    ### END CODE HERE ###
    print("Mean of each feature:", miu)
    print("Variance of each feature:", variances) 
    return miu, variances

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers based on the results from a validation set (p_val) 
    and the ground truth (y_val)

    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """
    print(f"\n=== {select_threshold.__name__} ===")
    #print(f"y_val: {y_val.shape}, p_val: {p_val.shape}")
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        ### START CODE HERE ### 
        predictions = p_val < epsilon
        F1 = F1Score(y_val, predictions)
        ### END CODE HERE ### 
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1

def multivariate_gaussian(X, mu, var):
    """
    Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """
    k = len(mu)
    if var.ndim == 1:
        var = np.diag(var)
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    return p

if __name__ == "__main__":
    # Load the dataset
    X_train = np.load("data/X_part1.npy")
    X_val = np.load("data/X_val_part1.npy")
    y_val = np.load("data/y_val_part1.npy")
    # Estimate mean and variance of each feature
    mu, var = estimate_gaussian(X_train)
    # Returns the density of the multivariate normal
    # at each data point (row) of X_train
    p_val = multivariate_gaussian(X_val, mu, var)
    #print(f"X_val: {X_val.shape}, var: {var.shape}, p_val: {p_val.shape}")
    epsilon, F1 = select_threshold(y_val, p_val)
    print('Best epsilon found using cross-validation: %e' % epsilon)
    print('Best F1 on Cross Validation Set: %f' % F1)

    X_train = np.load("data/X_part2.npy")
    X_val = np.load("data/X_val_part2.npy")
    y_val = np.load("data/y_val_part2.npy")
    print ('The shape of X_train_high is:', X_train.shape)
    print ('The shape of X_val_high is:', X_val.shape)
    print ('The shape of y_val_high is: ', y_val.shape)    
    # Apply the same steps to the larger dataset

    # Estimate the Gaussian parameters
    mu_high, var_high = estimate_gaussian(X_train)

    # Evaluate the probabilites for the training set
    p_high = multivariate_gaussian(X_train, mu_high, var_high)

    # Evaluate the probabilites for the cross validation set
    p_val_high = multivariate_gaussian(X_val, mu_high, var_high)

    # Find the best threshold
    epsilon_high, F1_high = select_threshold(y_val, p_val_high)

    print('Best epsilon found using cross-validation: %e'% epsilon_high)
    print('Best F1 on Cross Validation Set:  %f'% F1_high)
    print('# Anomalies found: %d'% sum(p_high < epsilon_high))
