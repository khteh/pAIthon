import numpy
from Sigmoid import sigmoid
from Softmax import softmax
"""
Input with n_x number of units
For a single time step of a single input example, x(i)(t) is a one-dimensional input vector
Using language as an example, a language with a 5000-word vocabulary could be one-hot encoded into a vector that has 5000 units. So x(i)(t) would have the shape (5000,)
The notation n_x is used here to denote the number of units in a single time step of a single training example. 
n_x = len(x(i)) with x(i) is a single input sample, every sample one-hot encoded.

Time steps of size T_x
A recurrent neural network has multiple time steps, which you'll index with t.
For example, a single training example x(i) consisting of multiple time steps T_x. Here, T_x will denote the number of timesteps in the longest sequence.

Batches of size m
Let's say we have mini-batches, each with 20 training examples
To benefit from vectorization, you'll stack 20 columns of x(i) examples
For example, this tensor has the shape (5000,20,10) with 5000 words, 20 training samples, 10 time steps (T_x)
You'll use m to denote the number of training examples. So, the shape of a mini-batch is (n_x,m,T_x)
"""
def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    print(f"Waa: {Waa.shape}, a_prev: {a_prev.shape}, Wax: {Wax.shape}, xt: {xt.shape}, ba: {ba.shape}")

    # compute next activation state using the formula given above
    # a(t) = g(Waa * a(t-1) + Wax * x(t) + ba)
    # at = g(Wa*[a(t-1), x(t)] + ba)
    a_next = numpy.tanh(Waa @ a_prev + Wax @ xt + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(Wya @ a_next + by)
    
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: rnn_forward

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    print(f"x: {x.shape}")
    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    ### START CODE HERE ###
    
    # initialize "a" and "y_pred" with zeros (≈2 lines)
    a = numpy.zeros((n_a, m, T_x))
    y_pred = numpy.zeros((n_y, m, T_x))
    
    # Initialize a_next (≈1 line)
    a_next = a0
    
    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache (≈1 line)
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[:,:,t] = yt_pred
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)
    ### END CODE HERE ###
    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    ### START CODE HERE ###
    # Concatenate a_prev and xt (≈1 line)
    # Wa is a matrix of [Waa | Wax] stacked horizontally with shape (100, 10100)
    # [a(t-1, x(t))] is a matrix of [a(t-1) | x(t)] stacked vertically with shape (10100, 1)
    concat = numpy.concatenate((a_prev, xt), axis=0) # horizontally stacked

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    """
    cct: candidate value C~(t) = tanh(Wc[a(t-1), x(t)] + bc)
    
    Wi is the update gate weight
    bi is the update gate bias
    it is the update gate
    
    Wo: output gate weight
    bo: output gate bias
    ot: output gate
    
    Wf: forget gate weight
    bf: forget gate bias
    ft: forget gate
    - Update Gate [0, 1], Gu = sigmoid(Wc @ [a(t-1), x(t)] + bu) : 1: Update 0: Don't update
    - Forget Gate [0, 1], Gf = sigmoid(Wf @ [a(t-1), x(t)] + bf) : 1: Forget 0: Don't forget
    - Output Gate [0, 1], Go = sigmoid(Wo @ [a(t-1), x(t)] + bo) : 1: Forget 0: Don't forget
    - C(t) = Gu * C~(t) + Gf * C(t-1) : element-wise multiplications.
    - a(t) = Go * tanh(C(t))
    """
    ft = sigmoid(Wf @ concat + bf) # forget
    it = sigmoid(Wi @ concat + bi) # update
    cct = numpy.tanh(Wc @ concat + bc) # candidate value
    c_next = it * cct + ft * c_prev
    ot = sigmoid(Wo @ concat + bo) # output gate
    a_next = ot * numpy.tanh(c_next)
    
    # Compute prediction of the LSTM cell (≈1 line)
    # y^(t) = g(Wya @ a(t) + by)
    yt_pred = softmax(Wy @ a_next + by)
    ### END CODE HERE ###

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []
    
    ### START CODE HERE ###
    Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape
    
    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = numpy.zeros((n_a, m, T_x))
    c = numpy.zeros((n_a, m, T_x))
    y = numpy.zeros((n_y, m, T_x))
    
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = numpy.zeros((n_a, m))
    
    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:,:,t]
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(x[:,:,t], a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the next cell state (≈1 line)
        c[:,:,t]  = c_next
        # Save the value of the prediction in y (≈1 line)
        y[:,:,t] = yt
        # Append the cache into caches (≈1 line)
        caches.append(cache)
        
    caches = (caches, x)
    return a, y, c, caches

def rnn_cell_forward_tests():
    xt_tmp = numpy.random.randn(3, 10)
    a_prev_tmp = numpy.random.randn(5, 10)
    parameters_tmp = {}
    parameters_tmp['Waa'] = numpy.random.randn(5, 5)
    parameters_tmp['Wax'] = numpy.random.randn(5, 3)
    parameters_tmp['Wya'] = numpy.random.randn(2, 5)
    parameters_tmp['ba'] = numpy.random.randn(5, 1)
    parameters_tmp['by'] = numpy.random.randn(2, 1)

    a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
    print("a_next[4] = \n", a_next_tmp[4])
    print("a_next.shape = \n", a_next_tmp.shape)
    print("yt_pred[1] =\n", yt_pred_tmp[1])
    print("yt_pred.shape = \n", yt_pred_tmp.shape)    

def rnn_forward_tests():
    x_tmp = numpy.random.randn(3, 10, 4)
    a0_tmp = numpy.random.randn(5, 10)
    parameters_tmp = {}
    parameters_tmp['Waa'] = numpy.random.randn(5, 5)
    parameters_tmp['Wax'] = numpy.random.randn(5, 3)
    parameters_tmp['Wya'] = numpy.random.randn(2, 5)
    parameters_tmp['ba'] = numpy.random.randn(5, 1)
    parameters_tmp['by'] = numpy.random.randn(2, 1)

    a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
    print("a[4][1] = \n", a_tmp[4][1])
    print("a.shape = \n", a_tmp.shape)
    print("y_pred[1][3] =\n", y_pred_tmp[1][3])
    print("y_pred.shape = \n", y_pred_tmp.shape)
    print("caches[1][1][3] =\n", caches_tmp[1][1][3])
    print("len(caches) = \n", len(caches_tmp))

def lstm_cell_forward_tests():
    xt_tmp = numpy.random.randn(3, 10)
    a_prev_tmp = numpy.random.randn(5, 10)
    c_prev_tmp = numpy.random.randn(5, 10)
    parameters_tmp = {}
    parameters_tmp['Wf'] = numpy.random.randn(5, 5 + 3)
    parameters_tmp['bf'] = numpy.random.randn(5, 1)
    parameters_tmp['Wi'] = numpy.random.randn(5, 5 + 3)
    parameters_tmp['bi'] = numpy.random.randn(5, 1)
    parameters_tmp['Wo'] = numpy.random.randn(5, 5 + 3)
    parameters_tmp['bo'] = numpy.random.randn(5, 1)
    parameters_tmp['Wc'] = numpy.random.randn(5, 5 + 3)
    parameters_tmp['bc'] = numpy.random.randn(5, 1)
    parameters_tmp['Wy'] = numpy.random.randn(2, 5)
    parameters_tmp['by'] = numpy.random.randn(2, 1)

    a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)

    print("a_next[4] = \n", a_next_tmp[4])
    print("a_next.shape = ", a_next_tmp.shape)
    print("c_next[2] = \n", c_next_tmp[2])
    print("c_next.shape = ", c_next_tmp.shape)
    print("yt[1] =", yt_tmp[1])
    print("yt.shape = ", yt_tmp.shape)
    print("cache[1][3] =\n", cache_tmp[1][3])
    print("len(cache) = ", len(cache_tmp))

def lstm_forward_tests():
    x_tmp = numpy.random.randn(3, 10, 7)
    a0_tmp = numpy.random.randn(5, 10)
    parameters_tmp = {}
    parameters_tmp['Wf'] = numpy.random.randn(5, 5 + 3)
    parameters_tmp['bf'] = numpy.random.randn(5, 1)
    parameters_tmp['Wi'] = numpy.random.randn(5, 5 + 3)
    parameters_tmp['bi']= numpy.random.randn(5, 1)
    parameters_tmp['Wo'] = numpy.random.randn(5, 5 + 3)
    parameters_tmp['bo'] = numpy.random.randn(5, 1)
    parameters_tmp['Wc'] = numpy.random.randn(5, 5 + 3)
    parameters_tmp['bc'] = numpy.random.randn(5, 1)
    parameters_tmp['Wy'] = numpy.random.randn(2, 5)
    parameters_tmp['by'] = numpy.random.randn(2, 1)

    a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
    print("a[4][3][6] = ", a_tmp[4][3][6])
    print("a.shape = ", a_tmp.shape)
    print("y[1][4][3] =", y_tmp[1][4][3])
    print("y.shape = ", y_tmp.shape)
    print("caches[1][1][1] =\n", caches_tmp[1][1][1])
    print("c[1][2][1]", c_tmp[1][2][1])
    print("len(caches) = ", len(caches_tmp))

if __name__ == "__main__":
    rnn_cell_forward_tests()
    rnn_forward_tests()
    lstm_cell_forward_tests()
    lstm_forward_tests()