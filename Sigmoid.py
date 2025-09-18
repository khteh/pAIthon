import numpy, tensorflow as tf

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
    return 1.0 / (1.0 + numpy.exp(-z))

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

if __name__ == "__main__":
    assert tf.math.sigmoid(123.456) == sigmoid(123.456)
    input = numpy.array([123.456, 789.012])
    assert (tf.math.sigmoid(input).numpy() == sigmoid(input)).all()
    x = numpy.array([1, 2, 3])
    # [0.19661193 0.10499359 0.04517666]
    expected_output = numpy.array([0.19661193,
                                0.10499359,
                                0.04517666])
    derivatives = sigmoid_derivative(x)
    print(f"sigmoid derivatives: {derivatives}")
    #assert (expected_output == derivatives).all() rounding error?