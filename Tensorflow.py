import numpy, tensorflow as tf
from tensorflow.keras.optimizers import Adam

"""
Use tf to optimize J(w) = w**2 - 10w + 25 = (w - 5) ** 2
So, w = 5 to optimize J(w) to the minimum of 0.
"""

class Basic():
    _w = None
    _x = None
    _optimizer = None
    _trainable_vars = None
    _epochs:int = None

    def __init__(self, epochs: int):
        self._epochs = epochs
        self._w = tf.Variable(0, dtype=tf.float64)
        self._x = numpy.array([1, -10, 25], dtype=numpy.float64) # This is the coefficients of the w parameter in the cost function
        self._optimizer = Adam(0.1)
        self._trainable_vars = [self._w]

    def _cost(self):
        return self._x[0] * self._w ** 2 + self._x[1] * self._w + self._x[2] # Forward propagation. TF will compute the computation graph of this.
    
    def _train_step(self):
        with tf.GradientTape() as tape:
            loss = self._cost()
        gradients = tape.gradient(loss, self._trainable_vars)
        self._optimizer.apply_gradients(zip(gradients, self._trainable_vars))

    def Train(self):
        for i in range(self._epochs):
            self._train_step()
        return self._w

def NumpyChannelLast():
    batch_size = 1
    image_channels = 2
    image_size = 3
    data = numpy.ones((batch_size, image_channels, image_size, image_size))
    data[:, :, 0] = 0
    print(f"data: {data.shape} {data}")
    data_transposed = tf.transpose(data, perm=[0,2,3,1])
    print(f"data_transposed: {type(data_transposed)} {data_transposed.shape} {data_transposed}")

if __name__ == "__main__":
    simple = Basic(1000)
    w = simple.Train()
    assert numpy.allclose(w, 5.0), f"Unexpected gradients. Expect 5.0 but get {w}"
    NumpyChannelLast()