import numpy, tensorflow as tf

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    A = numpy.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = numpy.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z (dJ/dz)
    """
    Z = cache
    A = sigmoid(Z)
    dZ = dA * A * (1-A)
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid(Z):
    """
    Compute the sigmoid of z (logit). 
    Sigmoid is for binary or multi-label classification, producing independent probabilities for each class.

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    Recall that for logistic regression, the model is represented as
    f(w,b) = g(w.x + b)
 
    where function g is the sigmoid function. The sigmoid function is defined as:
    g(z) = 1 / (1 + exp(-z))
    """
    Z = numpy.clip(Z, -500, 500 )           # protect against overflow
    return 1.0 / (1.0 + numpy.exp(-Z))

def sigmoid_derivative(x):
    """
    - Let a = y^
        (x1,w1,x2,w2,b) => z = x1w1 + x2w2 + b => a = sigmoid(z) => L(a,y) (Forward propagation)
                        dL/dz = dL/da * da/dz  <= d(L)/da = - y/a + (1-y)/(1-a) (Backward propagation) Note: d (ln(a)) / da = 1 / a
                        da/dz = a(1-a)
                        dL/dz = a - y
    """
    s = sigmoid(x)
    return s * (1 - s)

def softmax(z):
    """
    Compupte the softmax of z (logit).
    Softmax is for multi-class classification, transforming outputs into a probability distribution where probabilities sum to one and are dependent on each other. Softmax is essentially a generalized version of Sigmoid.

    Args:
        z (ndarray): A scalar, numpy array of any size.
    
    Returns:
        g (ndarray): softmax(z). with the same shape as z

    Recall that for logistic regression, the model is represented as
    f(w,b) = g(w.x + b)
 
    where function g is the softmax function. The softmax function is defined as:
    g(z) = e^Z / sum(e^Z)

    Let Z = x - max(x)
    e^(x - max(x)) = e^x * e^-max(x) = e^x  / e^max(x)
    g(e^(x - max(x))) = (e^x  / e^max(x)) / sum(e^x  / e^max(x)) = (e^x  / e^max(x)) / ((1  / e^max(x)) * sum(e^x)) = ((e^x  / e^max(x)) / sum(e^x)) * e^max(x) = e^x / sum(e^x)
    So, softmax(x) = softmax(1 - whatever)
    """
    try:
        ez = numpy.exp(z - numpy.max(z)) # Element-size exponential. math.exp won't work as it expects scalar input parameter
        return ez / numpy.sum(ez)
    except OverflowError as e:
        print(f"Overflow! {e}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    assert tf.math.sigmoid(123.456) == sigmoid(123.456)
    input = numpy.array([123.456, 789.012])
    A, cache = sigmoid(input)
    assert (tf.math.sigmoid(input).numpy() == sigmoid(input)).all()
    assert (A == cache).all()
    x = numpy.array([1, 2, 3])
    # [0.19661193 0.10499359 0.04517666]
    expected_output = numpy.array([0.19661193,
                                0.10499359,
                                0.04517666])
    derivatives = sigmoid_derivative(x)
    print(f"sigmoid derivatives: {derivatives}")
    #assert (expected_output == derivatives).all() rounding error?