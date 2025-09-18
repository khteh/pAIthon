import numpy
"""
Types of loss functions:
0-1 Loss: Used in discrete classification
L1 Loss: |actual - prediction| -> Used in continuous number prediction. Used when we don't care about outliers.
L2 Loss: (actual - prediction)^2 -> Penalizes worse / bigger loss more harshly. Used when we care about outliers.
"""
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    return numpy.sum(numpy.abs(yhat - y))

def L2(yhat, y):
    return numpy.sum(numpy.abs(yhat - y) ** 2)

if __name__ == "__main__":
    yhat = numpy.array([.9, 0.2, 0.1, .4, .9])
    y = numpy.array([1, 0, 0, 1, 1])
    l1 = L1(yhat, y)
    print(f"L1 = {l1}")
    assert 1.1 == l1

    yhat = numpy.array([.9, 0.2, 0.1, .4, .9])
    y = numpy.array([1, 0, 0, 1, 1])
    l2 = L2(yhat, y)
    print(f"L2 = {l2}")
    assert 0.43 == l2
