import numpy, math, tensorflow as tf

def sigmoid(z):
    """
    Compute the sigmoid of z. 
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
    z = numpy.clip( z, -500, 500 )           # protect against overflow
    return 1.0/(1.0+numpy.exp(-z))

if __name__ == "__main__":
    assert tf.math.sigmoid(123.456) == sigmoid(123.456)
    input = numpy.array([123.456, 789.012])
    assert (tf.math.sigmoid(input).numpy() == sigmoid(input)).all()