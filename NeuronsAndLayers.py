import numpy, warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import layers, losses, optimizers, regularizers
warnings.simplefilter(action='ignore', category=FutureWarning)

X_train = numpy.array([[1.0], [2.0]], dtype=numpy.float32)           #(size in 1000 square feet)
Y_train = numpy.array([[300.0], [500.0]], dtype=numpy.float32)       #(price in 1000s of dollars)

def LinearRegressionModel():
    """
    Neuron without activation - Regression/Linear Model
    """
    # Data set
    X_train = numpy.array([[1.0], [2.0]], dtype=numpy.float32)           #(size in 1000 square feet)
    Y_train = numpy.array([[300.0], [500.0]], dtype=numpy.float32)       #(price in 1000s of dollars)
    fig, ax = plt.subplots(1,1)
    ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
    ax.legend( fontsize='xx-large')
    ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
    ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
    plt.show()
    """
    The function implemented by a neuron with no activation is the same as linear regression:
    f_w_b(x) = numpy.dot(w, x) + b
    We can define a layer with one neuron or unit and compare it to the familiar linear regression function.
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.      
    """       
    linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', name="L1", kernel_regularizer=regularizers.l2(0.01)) # Decrease to fix high bias; Increase to fix high variance.
    w, b = linear_layer.get_weights()
    # There are no weights as the weights are not yet instantiated. 
    print(f"weights: {w}")
    # Let's try the model on one example in X_train. This will trigger the instantiation of the weights.
    # Note, the input to the layer must be 2-D, so we'll reshape it.
    a1 = linear_layer(X_train[0].reshape(1,1))
    # The result is a tensor (another name for an array) with a shape of (1,1) or one entry.
    # These weights are randomly initialized to small numbers and the bias defaults to being initialized to zero.
    print(f"a1: {a1}")
    w, b = linear_layer.get_weights()
    # A linear regression model (1) with a single input feature will have a single weight and bias. This matches the dimensions of our `linear_layer` above. 
    print(f"w = {w}, b={b}")
    # The weights are initialized to random values so let's set them to some known values.
    set_w = numpy.array([[200]])
    set_b = numpy.array([100])
    # set_weights takes a list of numpy arrays
    linear_layer.set_weights([set_w, set_b])
    w, b = linear_layer.get_weights()
    # Manual calculation of parameters and bias:
    a1 = linear_layer(X_train[0].reshape(1,1))
    print(f"Neuron output: {a1}")
    alin = numpy.dot(set_w,X_train[0].reshape(1,1)) + set_b
    print(f"Linear Regression output: {alin}")
    assert (a1.numpy() == alin).all()
    prediction_tf = linear_layer(X_train)
    prediction_np = numpy.dot( X_train, set_w) + set_b
    assert (prediction_tf.numpy() == prediction_np).all()

def LogisticNeuron():
    """
    Neuron with Sigmoid activation.
    We can implement a 'logistic neuron' by adding a sigmoid activation. Use Tensorflow Sequential model.
    """
    # Data set
    X_train = numpy.array([0., 1, 2, 3, 4, 5], dtype=numpy.float32).reshape(-1,1)  # 2-D Matrix
    Y_train = numpy.array([0,  0, 0, 1, 1, 1], dtype=numpy.float32).reshape(-1,1)  # 2-D Matrix
    pos = Y_train == 1
    neg = Y_train == 0

    fig,ax = plt.subplots(1,1,figsize=(4,3))
    ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none', edgecolors="dlblue",lw=3)
    ax.set_ylim(-0.08,1.1)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('one variable plot')
    ax.legend(fontsize=12)
    plt.show()
    """
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.      
    """
    model = Sequential(
        [
            tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1', kernel_regularizer=regularizers.l2(0.01)) # Decrease to fix high bias; Increase to fix high variance.
        ]
    )
    # model.summary() shows the layers and number of parameters in the model. There is only one layer in this model and that layer has only one unit. The unit has two parameters, w and b
    # The weights  𝑊 should be of size (number of features in input, number of units in the layer) while the bias b size should match the number of units in the layer.
    model.summary()
    logistic_layer = model.get_layer('L1')
    w,b = logistic_layer.get_weights()
    print(f"w: {w.shape} {w}, b: {b.shape} {b}")
    set_w = numpy.array([[2]])
    set_b = numpy.array([-4.5])
    # set_weights takes a list of numpy arrays
    logistic_layer.set_weights([set_w, set_b])
    print(f"Logistic Regression weights: {logistic_layer.get_weights()}")
    a1 = model.predict(X_train[0].reshape(1,1))
    print(f"Neuron output: {a1}")
    #alog = sigmoidnp(numpy.dot(set_w,X_train[0].reshape(1,1)) + set_b)
    #print(f"Linear Regression output: {alog}")

def TensorFlowDataNormalization(X):
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    return norm_l(X)

def Dense(a_in, W, b):
    """
    Demonstrate manual implementation of a single layer of neurons. #neurons determined by the W.shape[1], i.e., #columns in W
    W = numpy.array([
                [1, -3, 5],
                [2, 4, -6]
            ])
    w[1,1] = [1,2] layer 1, neuron 1
    w[1,2] = [-3, 4] layer 1, neuron 2
    w[1,3] = [5, -6] layer 1, neuron 3
    """
    neurons = W.shape[1] # columns
    a_out = numpy.zeros(neurons)
    for i in range(neurons):
        w = W[:, i] # Pull out the column values
        z = numpy.dot(w, a_in) + b[i]
        a_out[i] = sigmoid(z)
    return a_out

def DenseVectorized(a_in, W, b, g):
    """
    Demonstrate vectorized manual implementation of a single layer of neurons. #neurons determined by the W.shape[1], i.e., #columns in W
    W = numpy.array([
                [1, -3, 5],
                [2, 4, -6]
            ])
    w[1,1] = [1,2] layer 1, neuron 1
    w[1,2] = [-3, 4] layer 1, neuron 2
    w[1,3] = [5, -6] layer 1, neuron 3
    """
    return g(a_in @ W + b)

def Sequential(x, parameters: list[tuple]):
    """
    Demonstrate manual implementation of a TF NN by strining together multiple dense layers
    """
    result = x
    for i in parameters:
        result = DenseVectorized(result, i[0], i[1], sigmoid)
    return result

def Predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    predictions = numpy.zeros((m,1))
    for i in range(m):
        predictions[i,0] = Sequential(X[i], W1, b1, W2, b2)
    return  predictions

if __name__ == "__main__":
    X_tst = numpy.array([
        [200,13.9],  # postive example
        [200,17]])   # negative example
    X_tstn = TensorFlowDataNormalization(X_tst)  # remember to normalize
    W1_tmp = numpy.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
    b1_tmp = numpy.array( [-9.82, -9.28,  0.96] )
    W2_tmp = numpy.array( [[-31.18], [-27.59], [-32.56]] )
    b2_tmp = numpy.array( [15.41] )    
    predictions = Predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
    categories = (predictions >= 0.5).astype(int)
    print(f"Predictions: {predictions}, {categories}")
