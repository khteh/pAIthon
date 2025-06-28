import logging, numpy, math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
from matplotlib.widgets import Slider
from tensorflow.keras import layers, losses, optimizers, regularizers
"""
SparseCategorialCrossentropy or CategoricalCrossEntropy
Tensorflow has two potential formats for target values and the selection of the loss defines which is expected.

SparseCategorialCrossentropy: expects the target/label to be an integer corresponding to the index. For example, if there are 10 potential target values, y would be between 0 and 9.
CategoricalCrossEntropy: Expects the target/label value of an example to be one-hot encoded where the value at the target index is 1 while the other N-1 entries are zero. An example with 10 potential target values, where the target is 2 would be [0,0,1,0,0,0,0,0,0,0].    
"""
def softmax(z):
    """
    Compupte the softmax of z

    Args:
        z (ndarray): A scalar, numpy array of any size.
    
    Returns:
        g (ndarray): softmax(z). with the same shape as z

    Recall that for logistic regression, the model is represented as
    f(w,b) = g(w.x + b)
 
    where function g is the softmax function. The softmax function is defined as:
    g(z) = e^Z / sum(e^Z)
    """
    ez = numpy.exp(z) # Element-size exponential. math.exp won't work as it expects scalar input parameter
    return ez / numpy.sum(ez)

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
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.
    """
    model = Sequential(
        [ 
            Dense(25, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance.
            Dense(15, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(4, activation = 'softmax')    # < softmax activation here
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
    )
    model.fit(
        X_train,y_train,
        epochs=10
    )
    # Because the softmax is integrated into the output layer, the output is a vector of probabilities.
    p_nonpreferred = model.predict(X_train)
    print(p_nonpreferred.head())
    print("largest value", numpy.max(p_nonpreferred), "smallest value", numpy.min(p_nonpreferred))

def NNStableSoftmax(X_train, y_train):
    """
    More stable and accurate results can be obtained if the sigmoid/softmax and loss are combined during training.
    In the preferred organization the final layer has a linear activation. For historical reasons, the outputs in this form are referred to as *logits*. The loss function has an additional argument: `from_logits = True`. This informs the loss function that the sigmoid/softmax operation should be included in the loss calculation. This allows for an optimized implementation.
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.
    """
    preferred_model = Sequential(
        [ 
            Dense(25, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance.
            Dense(15, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(4, activation = 'linear')   #<-- Note
        ]
    )
    preferred_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
        optimizer=tf.keras.optimizers.Adam(0.001), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
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

if __name__ == "__main__":
    X_train, y_train = PrepareData()
    NNSoftmax(X_train, y_train)
    NNStableSoftmax(X_train, y_train)