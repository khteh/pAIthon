import numpy, math, tensorflow as tf

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    Recall that for logistic regression, the model is represented as
    f(w,b) = g(w.x + b)
 
    where function g is the sigmoid function. The sigmoid function is defined as:
    g(z) = 1 / (1 + exp(-z))
    """
    if numpy.isscalar(z):
        return 1 / (1 + math.exp(-z))
    elif isinstance(z, numpy.ndarray):
        #print(f"input: {z}, shape: {z.shape}, ndim: {z.ndim}")
        g = numpy.zeros_like(z)
        if z.ndim == 1:
            for i in range(len(z)):
                g[i] = 1 / (1 + math.exp(-z[i]))
        elif z.ndim == 2:
            rows, cols = z.shape
            for r in range(rows):
                for c in range(cols):
                    g[r][c] = 1 / (1 + math.exp(-z[r][c]))
            #print(f"z: {i}, result: {result}, g: {g}")
    #return numpy.array(g)
    return g

if __name__ == "__main__":
    assert tf.math.sigmoid(123.456) == sigmoid(123.456)
    input = numpy.array([123.456, 789.012])
    assert tf.math.sigmoid(input).numpy().all() == sigmoid(input).all()