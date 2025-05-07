import numpy
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64DXSM
# https://realpython.com/python-ai-neural-network/
rng = Generator(PCG64DXSM())

class NeuralNetwork():
    _weights: numpy.array = None
    _bias = None
    _rate = None
    def __init__(self, learning_rate):
        self._rate = learning_rate
        self._weights = [rng.randn(), rng.randn()]
        self._bias = rng.randn()

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def SigmoidDerivative(self, x):
        """
        You can take the derivative of the sigmoid function by multiplying sigmoid(x) and 1 - sigmoid(x).
        """
        tmp = self.sigmoid(x)
        return tmp * (1 - tmp)

    def Predict(self, input):
        l1 = numpy.dot(input, self._weights) + self._bias
        l2 = self.sigmoid(l1)
        return l2

    def ComputeGradients(self, input, target):
        """
        The power rule states that the derivative of xⁿ is nx⁽ⁿ⁻¹⁾. So the derivative of np.square(x) is 2 * x, and the derivative of x is 1.
        """
        l1 = numpy.dot(input, self._weights) + self._bias
        l2 = self.sigmoid(l1)
        prediction = l2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dl1 = self.SigmoidDerivative(l1)
        dl1_dbias = 1
        dl1_dweights = (0 * self._weights) + (1 * input)
        derror_bias = derror_dprediction * dprediction_dl1 * dl1_dbias
        derror_weights = derror_dprediction * dprediction_dl1 * dl1_dweights
        return derror_bias, derror_weights
    
    def UpdateParameters(self, derror_dbias, derror_dweights):
        self._bias -= derror_dbias * self._rate
        self._weights -= derror_dweights * self._rate

    def Train(self, input_vectors, targets, iterations):
        cummulative_errors = []
        for i in range(iterations):
            # Randomly pick a data point
            index = rng.randint(len(input_vectors))
            input = input_vectors[index]
            target = targets[index]

            # Compute gradients and update the weights
            self.UpdateParameters(self.ComputeGradients(input, target))
            if not i % 100:
                cummulative_error = 0
                # Loop through all the instances to measure the error
                for j in range(len(input_vectors)):
                    d = input_vectors[j]
                    target = targets[j]
                    prediction = self.Predict(d)
                    error = numpy.square(prediction - target) # Mean Squared Error
                    cummulative_error += error
                cummulative_errors.append(cummulative_error)
        return cummulative_errors

def Predict():
    input_vectors = numpy.array(
         [
             [3, 1.5],
             [2, 1],
             [4, 1.5],
             [3, 4],
             [3.5, 0.5],
             [2, 0.5],
             [5.5, 1],
             [1, 1],
         ]
     )
    targets = numpy.array([0, 1, 0, 1, 0, 1, 1, 0])
    neural_network = NeuralNetwork(0.1)
    training_error = neural_network.Train(input_vectors, targets, 10000)
    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.savefig("/tmp/cumulative_error.png")

if __name__ == "__main__":
    Predict()