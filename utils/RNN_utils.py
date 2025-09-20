import numpy
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)

def get_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    return txt

def get_initial_loss(vocab_size, seq_length):
    return -numpy.log(1.0/vocab_size)*seq_length

def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values
    
    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    numpy.random.seed(1)
    Wax = rng.standard_normal((n_a, n_x)) * numpy.sqrt(2/n_x) # input to hidden
    Waa = rng.standard_normal((n_a, n_a)) * numpy.sqrt(2/n_a) # hidden to hidden
    Wya = rng.standard_normal((n_y, n_a)) * numpy.sqrt(2/n_a) # hidden to output
    b = numpy.zeros((n_a, 1)) # hidden bias
    by = numpy.zeros((n_y, 1)) # output bias
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = numpy.tanh((Wax @ x) + (Waa @ a_prev) + b) # hidden state. Similar to sigmoid graph but the output is [-1, 1]
    p_t = softmax((Wya @ a_next) + by) # unnormalized log probabilities for next chars # probabilities for next chars 
    return a_next, p_t

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    gradients['dWya'] += dy @ a.T
    gradients['dby'] += dy
    da = (parameters['Wya'].T @ dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += daraw @ x.T
    gradients['dWaa'] += daraw @ a_prev.T
    gradients['da_next'] = parameters['Waa'].T @ daraw
    return gradients

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def rnn_forward(X, Y, a0, parameters, vocab_size = 27):
    
    # Initialize x, a and y_hat as empty dictionaries
    x, a, y_hat = {}, {}, {}
    
    a[-1] = numpy.copy(a0)
    
    # initialize your loss to 0
    loss = 0
    
    for t in range(len(X)):
        
        # Set x[t] to be the one-hot vector representation of the t'th character in X.
        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector. 
        x[t] = numpy.zeros((vocab_size,1)) 
        if (X[t] != None):
            x[t][X[t]] = 1
        
        # Run one step forward of the RNN
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        
        # Update the loss by substracting the cross-entropy term of this time-step from it.
        loss -= numpy.log(y_hat[t][Y[t],0])
        
    cache = (y_hat, a, x)
        
    return loss, cache

def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}
    
    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = numpy.zeros_like(Wax), numpy.zeros_like(Waa), numpy.zeros_like(Wya)
    gradients['db'], gradients['dby'] = numpy.zeros_like(b), numpy.zeros_like(by)
    gradients['da_next'] = numpy.zeros_like(a[0])
    
    ### START CODE HERE ###
    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = numpy.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    ### END CODE HERE ###
    
    return gradients, a
