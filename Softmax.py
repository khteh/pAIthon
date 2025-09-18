import logging, numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
"""
SparseCategorialCrossentropy or CategoricalCrossEntropy
Tensorflow has two potential formats for target values and the selection of the loss defines which is expected.

SparseCategorialCrossentropy: expects the target/label to be an integer corresponding to the index. For example, if there are 10 potential target values, y would be between 0 and 9.
CategoricalCrossEntropy: Expects the target/label value of an example to be one-hot encoded where the value at the target index is 1 while the other N-1 entries are zero. An example with 10 potential target values, where the target is 2 would be [0,0,1,0,0,0,0,0,0,0].    
"""
def softmax(z):
    """
    Compupte the softmax of z.
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

def PrepareData():
    # make  dataset for example
    coordinates = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
    X_train, y_train = make_blobs(n_samples=2000, centers=coordinates, cluster_std=1.0,random_state=30)
    # show classes in data set
    print(f"unique classes {numpy.unique(y_train)}")
    # show how classes are represented
    print(f"class representation {y_train[:10]}")
    # show shapes of our dataset
    print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")    
    return X_train, y_train

def NNSoftmax(X_train, y_train):
    """
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection by "turning off" less important features or nodes in the network.
                               Useful when there are many features and some might be irrelevant, as it can effectively perform feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero. This helps to prevent individual weights from becoming excessively large and dominating the model.
                               Generally preferred in deep learning for its ability to smoothly reduce weight magnitudes and improve model generalization without completely removing features.
    """
    print(f"\n=== {NNSoftmax.__name__} ===")
    model = Sequential(
        [
            Dense(25, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected
            Dense(15, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(4)   # Linear activation ("pass-through") if not specified
        ]
    )
    model.compile(
        loss=SparseCategoricalCrossentropy(), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
        optimizer=Adam(0.001), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
    )
    model.fit(
        X_train,y_train,
        epochs=10
    )
    # Because the softmax is integrated into the output layer, the output is a vector of probabilities.
    p_nonpreferred = model.predict(X_train)
    print(f"two example output vectors:\n {p_nonpreferred[:2]}")
    print("largest value", numpy.max(p_nonpreferred), "smallest value", numpy.min(p_nonpreferred))
    # To select the most likely category, the softmax is not required. One can find the index of the largest output using np.argmax().
    for i in range(5):
        print( f"{p_nonpreferred[i]}, category: {numpy.argmax(p_nonpreferred[i])}")

def NNStableSoftmax(X_train, y_train):
    """
    More stable and accurate results can be obtained if the sigmoid/softmax and loss are combined during training.
    In the preferred organization the final layer has a linear activation. For historical reasons, the outputs in this form are referred to as *logits*. The loss function has an additional argument: `from_logits = True`. This informs the loss function that the sigmoid/softmax operation should be included in the loss calculation. This allows for an optimized implementation.
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection by "turning off" less important features or nodes in the network.
                               Useful when there are many features and some might be irrelevant, as it can effectively perform feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero. This helps to prevent individual weights from becoming excessively large and dominating the model.
                               Generally preferred in deep learning for its ability to smoothly reduce weight magnitudes and improve model generalization without completely removing features.
    """
    print(f"\n=== {NNStableSoftmax.__name__} ===")
    preferred_model = Sequential(
        [ 
            Dense(25, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance.
            Dense(15, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(4, activation = 'linear') # Linear activation ("pass-through") if not specified
        ]
    )
    preferred_model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),  # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
        optimizer=Adam(0.001), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
    )
    preferred_model.fit(
        X_train,y_train,
        epochs=10
    )
    # Notice that in this model, the outputs are not probabilities, but can range from large negative numbers to large positive numbers. The output must be sent through a softmax when performing a prediction that expects a probability. 
    p_preferred = preferred_model.predict(X_train)
    print(f"two example output vectors:\n {p_preferred[:2]}")
    print("largest value", numpy.max(p_preferred), "smallest value", numpy.min(p_preferred))
    
    # The output predictions are NOT probabilities! If probabilities are the desired output, the output should be processed by a softmax.
    sm_preferred = tf.nn.softmax(p_preferred).numpy()
    print(f"two example output vectors:\n {sm_preferred[:2]}")
    print("largest value", numpy.max(sm_preferred), "smallest value", numpy.min(sm_preferred))
    # To select the most likely category, the softmax is not required. One can find the index of the largest output using np.argmax().
    for i in range(5):
        print( f"{p_preferred[i]}, category: {numpy.argmax(p_preferred[i])}")

def SoftmaxTests():
    v = numpy.array([333, 444, 555, 666, 777, 888, 999]).astype('float64')
    print(f"\n")
    print(f"softmax(v): {softmax(v)}")
    print(f"softmax(v - max(v)): {softmax(v - numpy.max(v))}")
    assert (softmax(v) == softmax(v - numpy.max(v))).all() # Assertion error

if __name__ == "__main__":
    X_train, y_train = PrepareData()
    NNSoftmax(X_train, y_train)
    NNStableSoftmax(X_train, y_train)
    SoftmaxTests()