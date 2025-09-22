import numpy, matplotlib.pyplot as plt, scipy.io, math, sklearn, sklearn.datasets
#from opt_utils_v1a import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
#from opt_utils_v1a import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from copy import deepcopy
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    numpy.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(numpy.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    inc = mini_batch_size

    # Step 2 - Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 64 examples.
    num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """
    L = len(parameters) // 2 # number of layers in the neural networks
    # Momentum update for each parameter
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v[f"dW{l}"]  + (1-beta) * grads[f"dW{l}"]
        v["db" + str(l)] = beta * v[f"db{l}"]  + (1-beta) * grads[f"db{l}"]
        # update parameters
        parameters["W" + str(l)] -= learning_rate * v[f"dW{l}"]
        parameters["b" + str(l)] -= learning_rate * v[f"db{l}"]
    return parameters, v

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using the Adam optimization algorithm.

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameter:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, maintained as a python dictionary
    s -- Adam variable, moving average of the squared gradient, maintained as a python dictionary
    t -- Adam variable, counts the number of steps taken for bias correction
    learning_rate -- learning rate, scalar
    beta1 -- exponential decay hyperparameter for the first moment estimates
    beta2 -- exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter to prevent division by zero in the Adam update, a small scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- updated Adam variable, moving average of the first gradient, python dictionary
    s -- updated Adam variable, moving average of the squared gradient, python dictionary
    v_corrected -- python dictionary containing bias-corrected first moment estimates
    s_corrected -- python dictionary containing bias-corrected second moment estimates
    """
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    # Perform Adam update on all parameters
    for l in range(1, L + 1):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l)] = beta1 * v[f"dW{l}"]  + (1-beta1) * grads[f"dW{l}"]
        v["db" + str(l)] = beta1 * v[f"db{l}"]  + (1-beta1) * grads[f"db{l}"]        
        
        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l)] = v[f"dW{l}"] / (1 - numpy.power(beta1, t))
        v_corrected["db" + str(l)] = v[f"db{l}"] / (1 - numpy.power(beta1, t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l)] = beta2 * s[f"dW{l}"]  + (1-beta2) * grads[f"dW{l}"]**2
        s["db" + str(l)] = beta2 * s[f"db{l}"]  + (1-beta2) * grads[f"db{l}"]**2
        
        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l)] = s[f"dW{l}"] / (1 - numpy.power(beta2, t))
        s_corrected["db" + str(l)] = s[f"db{l}"] / (1 - numpy.power(beta2, t))        
        
        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l)] -= learning_rate * v_corrected[f"dW{l}"] / (numpy.sqrt(s_corrected[f"dW{l}"]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected[f"db{l}"] / (numpy.sqrt(s_corrected[f"db{l}"]) + epsilon)
    return parameters, v, s, v_corrected, s_corrected

def random_mini_batches_tests():
    print(f"\n=== {random_mini_batches_tests.__name__} ===")
    #numpy.random.seed(1)
    mini_batch_size = 64
    nx = 12288
    m = 148
    X = numpy.array([x for x in range(nx * m)]).reshape((m, nx)).T
    Y = rng.standard_normal((1, m)) < 0.5

    mini_batches = random_mini_batches(X, Y, mini_batch_size)
    n_batches = len(mini_batches)

    assert n_batches == math.ceil(m / mini_batch_size), f"Wrong number of mini batches. {n_batches} != {math.ceil(m / mini_batch_size)}"
    for k in range(n_batches - 1):
        assert mini_batches[k][0].shape == (nx, mini_batch_size), f"Wrong shape in {k} mini batch for X"
        assert mini_batches[k][1].shape == (1, mini_batch_size), f"Wrong shape in {k} mini batch for Y"
        assert numpy.sum(numpy.sum(mini_batches[k][0] - mini_batches[k][0][0], axis=0)) == ((nx * (nx - 1) / 2 ) * mini_batch_size), "Wrong values. It happens if the order of X rows(features) changes"
    if ( m % mini_batch_size > 0):
        assert mini_batches[n_batches - 1][0].shape == (nx, m % mini_batch_size), f"Wrong shape in the last minibatch. {mini_batches[n_batches - 1][0].shape} != {(nx, m % mini_batch_size)}"

    assert numpy.allclose(mini_batches[0][0][0][0:3], [294912,  86016, 454656]), "Wrong values. Check the indexes used to form the mini batches"
    assert numpy.allclose(mini_batches[-1][0][-1][0:3], [1425407, 1769471, 897023]), "Wrong values. Check the indexes used to form the mini batches"

    mini_batch_size = 64
    X = rng.standard_normal((12288, 148))
    Y = rng.standard_normal((1, 148)) < 0.5
    mini_batches = random_mini_batches(X, Y, mini_batch_size)

    print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
    print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
    print("\033[92mAll tests passed!")

def update_parameters_with_momentum_tests():
    print(f"\n=== {update_parameters_with_momentum_tests.__name__} ===")
    W1 = rng.standard_normal((2,3))
    b1 = rng.standard_normal((2,1))
    W2 = rng.standard_normal((3,2))
    b2 = rng.standard_normal((3,1))

    dW1 = rng.standard_normal((2,3))
    db1 = rng.standard_normal((2,1))
    dW2 = rng.standard_normal((3,2))
    db2 = rng.standard_normal((3,1))
   
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    v = {'dW1': numpy.array([[ 0.,  0.,  0.],
                          [ 0.,  0.,  0.]]), 
         'dW2': numpy.array([[ 0.,  0.],
                          [ 0.,  0.],
                          [ 0.,  0.]]), 
         'db1': numpy.array([[ 0.],
                          [ 0.]]), 
         'db2': numpy.array([[ 0.],
                          [ 0.],
                          [ 0.]])}

    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
    print("W1 = \n" + str(parameters["W1"]))
    print("b1 = \n" + str(parameters["b1"]))
    print("W2 = \n" + str(parameters["W2"]))
    print("b2 = \n" + str(parameters["b2"]))
    print("v[\"dW1\"] = \n" + str(v["dW1"]))
    print("v[\"db1\"] = \n" + str(v["db1"]))
    print("v[\"dW2\"] = \n" + str(v["dW2"]))
    print("v[\"db2\"] = v" + str(v["db2"]))

def update_parameters_with_adam_tests():
    print(f"\n=== {update_parameters_with_adam_tests.__name__} ===")
    v, s = ({'dW1': numpy.array([[ 0.,  0.,  0.], # (2, 3)
                              [ 0.,  0.,  0.]]), 
             'dW2': numpy.array([[ 0.,  0.],      # (3, 2)
                              [ 0.,  0.],
                              [ 0.,  0.]]), 
             'db1': numpy.array([[ 0.],           # (2, 1)
                              [ 0.]]), 
             'db2': numpy.array([[ 0.],          # (3, 1)
                              [ 0.],
                              [ 0.]])}, 
            {'dW1': numpy.array([[ 0.,  0.,  0.], # (2, 3)
                              [ 0.,  0.,  0.]]), 
             'dW2': numpy.array([[ 0.,  0.],      # (3, 2)
                              [ 0.,  0.],
                              [ 0.,  0.]]), 
             'db1': numpy.array([[ 0.],           # (2, 1)
                              [ 0.]]), 
             'db2': numpy.array([[ 0.],           # (3, 1)
                              [ 0.],
                              [ 0.]])})
    W1 = rng.standard_normal((2,3))
    b1 = rng.standard_normal((2,1))
    W2 = rng.standard_normal((3,2))
    b2 = rng.standard_normal((3,1))

    dW1 = rng.standard_normal((2,3))
    db1 = rng.standard_normal((2,1))
    dW2 = rng.standard_normal((3,2))
    db2 = rng.standard_normal((3,1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    t = 2
    learning_rate = 0.02
    beta1 = 0.8
    beta2 = 0.888
    epsilon = 1e-2
    parameters, v, s, vc, sc  = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
    print(f"W1 = \n{parameters['W1']}")
    print(f"W2 = \n{parameters['W2']}")
    print(f"b1 = \n{parameters['b1']}")
    print(f"b2 = \n{parameters['b2']}")

if __name__ == "__main__":
    random_mini_batches_tests()
    update_parameters_with_momentum_tests()
    update_parameters_with_adam_tests()