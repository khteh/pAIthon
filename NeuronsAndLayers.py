import numpy, warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers, Sequential
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.regularizers import l2
from utils.GPU import InitializeGPU
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
    fig, ax = plt.subplots(1,1) # figsize = (width, height)
    ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
    ax.legend(fontsize='x-large')
    ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
    ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
    plt.show()
    """
    The function implemented by a neuron with no activation is the same as linear regression:
    f_w_b(x) = numpy.dot(w, x) + b
    We can define a layer with one neuron or unit and compare it to the familiar linear regression function.
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection by "turning off" less important features or nodes in the network.
                               Useful when there are many features and some might be irrelevant, as it can effectively perform feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero. This helps to prevent individual weights from becoming excessively large and dominating the model.
                               Generally preferred in deep learning for its ability to smoothly reduce weight magnitudes and improve model generalization without completely removing features.
    """       
    linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', name="L1", kernel_regularizer=l2(0.01)) # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected
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
    alin = (set_w @ X_train[0].reshape(1,1)) + set_b
    print(f"Linear Regression output: {alin}")
    assert (a1.numpy() == alin).all()
    prediction_tf = linear_layer(X_train)
    prediction_np = ( X_train @ set_w) + set_b
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

    fig, ax = plt.subplots(1,1,figsize=(4,3)) # figsize = (width, height)
    ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none', edgecolors="darkblue",lw=3)
    ax.set_ylim(-0.08,1.1)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('one variable plot')
    ax.legend(fontsize='x-large')
    plt.show()
    """
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection by "turning off" less important features or nodes in the network.
                               Useful when there are many features and some might be irrelevant, as it can effectively perform feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero. This helps to prevent individual weights from becoming excessively large and dominating the model.
                               Generally preferred in deep learning for its ability to smoothly reduce weight magnitudes and improve model generalization without completely removing features.
    """
    model = Sequential(
        [
            Input(shape=(1,)),
            tf.keras.layers.Dense(1, activation = 'sigmoid', name='L1', kernel_regularizer=l2(0.01)) # Decrease to fix high bias; Increase to fix high variance.
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
    #alog = sigmoidnp((set_w @ X_train[0].reshape(1,1)) + set_b)
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
        z = (w @ a_in) + b[i]
        a_out[i] = sigmoid(z)
    return a_out

def DenseVectorized(a_in, W, b, g = sigmoid):
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
    print(f"\n=== {DenseVectorized.__name__} ===")
    print(f"a_in: {a_in.shape}, W: {W.shape}")
    return g(a_in @ W + b)

def Predict(X, W: list[float], b: list[float]):
    """
    Demonstrate manual implementation of a TF NN by strining together multiple dense layers
    """
    assert len(W) == len(b)
    result = X
    for layer in range(len(W)):
        result = DenseVectorized(result, W[layer], b[layer], sigmoid)
    return result

def DenseVectorizedTest():
    print(f"\n=== {DenseVectorizedTest.__name__} ===")
    X_tst = 0.1*numpy.arange(1,9,1).reshape(4,2) # (4 examples, 2 features)
    W_tst = 0.1*numpy.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
    b_tst = 0.1*numpy.arange(1,4,1).reshape(1,3) # (1,3 features)
    A_tst = DenseVectorized(X_tst, W_tst, b_tst, sigmoid)
    print(f"A_tst: {A_tst}")

if __name__ == "__main__":
    InitializeGPU()
    LogisticNeuron()
    DenseVectorizedTest()
    X_tst = numpy.array([
        [200,13.9],  # postive example
        [200,17]])   # negative example
    X_tstn = TensorFlowDataNormalization(X_tst)  # remember to normalize
    W1_tmp = numpy.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
    b1_tmp = numpy.array( [-9.82, -9.28,  0.96] )
    W2_tmp = numpy.array( [[-31.18], [-27.59], [-32.56]] )
    b2_tmp = numpy.array( [15.41] )
    predictions = Predict(X_tstn, [W1_tmp, W2_tmp], [b1_tmp, b2_tmp])
    print(f"X: {X_tst}")
    print(f"Predictions: {predictions}")
    assert predictions[0][0] >= 0.5
    assert predictions[1][0] < 0.5