import numpy, copy, matplotlib.pyplot as plt, sklearn, sklearn.datasets, sklearn.linear_model
from Activations import sigmoid
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

class PlanarDataClassification():
    _X = None
    _Y = None
    _W1 = None # (n_h, n_x)
    _W2 = None # (n_y, n_h)
    _b1 = None # (n_h, 1)
    _b2 = None # (n_y, 1)
    _n_h:int = 4 # the size of the hidden layer
    _learning_rate:float = None
    _epochs:int = None

    def __init__(self, learning_rate:float, epochs:int):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._PrepareData()

    def BuildModel(self, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations

        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        # Loop (gradient descent)
        for i in range(0, self._epochs):
            
            #(≈ 4 lines of code)
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = self._forward_propagation()
            
            # Cost function. Inputs: "A2, Y". Outputs: "cost".
            cost = self._compute_cost(A2)
    
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = self._backward_propagation(cache)
    
            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            self._update_parameters(grads)
            
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

    def Predict(self, X = None):
        """
        Using the learned parameters, predicts a class for each example in X
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        X -- input data of size (n_x, m)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        if X is None:
            X = self._X
        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        #(≈ 2 lines of code)
        A2, cache = self._forward_propagation(X)
        predictions = A2 > 0.5
        #print(f"A2: {A2.shape} {A2}, _Y: {self._Y.shape}, predictions: {predictions.shape}")
        if self._Y.shape == predictions.shape:
            print ('Accuracy: %d' % float(((self._Y @ predictions.T) + ((1 - self._Y) @(1 - predictions.T))).item() / float(self._Y.size) * 100) + '%')
        return predictions

    def plot_decision_boundary(self):
        # Set min and max values and give it some padding
        x_min, x_max = self._X[0, :].min() - 1, self._X[0, :].max() + 1
        y_min, y_max = self._X[1, :].min() - 1, self._X[1, :].max() + 1
        h = 0.01
        print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
        # Generate a grid of points with distance h between them
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        model = lambda x: self.Predict(x.T)
        Z = model(numpy.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(self._X[0, :], self._X[1, :], c=self._Y, cmap=plt.cm.Spectral)
        plt.title(f"Decision Boundary for hidden layer size {self._n_h}")

    def TuneHiddenLayerSizes(self, sizes:int, epochs:int):
        plt.figure(figsize=(16, 32))
        #hidden_layer_sizes = [1, 2, 3, 4, 5]
        # you can try with different hidden layer sizes
        # but make sure before you submit the assignment it is set as "hidden_layer_sizes = [1, 2, 3, 4, 5]"
        # hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
        n_x = self._X.shape[0]
        n_y = self._Y.shape[0]
        self._epochs = epochs
        for i, n_h in enumerate(sizes):
            plt.subplot(5, 2, i+1)
            plt.title('Hidden Layer of size %d' % n_h)
            self._n_h = n_h
            self._W1 = rng.standard_normal((self._n_h, n_x)) * numpy.sqrt(2/n_x)
            self._b1 = numpy.zeros((self._n_h, 1))
            self._W2 = rng.standard_normal((n_y, self._n_h)) * numpy.sqrt(2/self._W1.shape[0])
            self._b2 = numpy.zeros((n_y, 1))
            self.BuildModel()
            self.plot_decision_boundary()
            predictions = self.Predict()
            accuracy = float(((self._Y @ predictions.T) + ((1 - self._Y) @(1 - predictions.T))).item() / float(self._Y.size) * 100)
            print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        m = 400 # number of examples
        N = int(m/2) # number of points per class
        D = 2 # dimensionality
        self._X = numpy.zeros((m,D)) # data matrix where each row is a single example
        self._Y = numpy.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
        a = 4 # maximum ray of the flower
        for j in range(2):
            ix = range(N*j,N*(j+1))
            t = numpy.linspace(j*3.12,(j+1)*3.12, N) + rng.standard_normal((1, N))*0.2 # theta
            r = a*numpy.sin(4*t) + rng.standard_normal((1, N))*0.2 # radius
            self._X[ix] = numpy.concatenate([r*numpy.sin(t), r*numpy.cos(t)]).T
            self._Y[ix] = j
        self._X = self._X.T
        self._Y = self._Y.T
        n_x = self._X.shape[0]
        n_y = self._Y.shape[0]
        self._W1 = rng.standard_normal((self._n_h, n_x)) * numpy.sqrt(2/n_x)
        self._b1 = numpy.zeros((self._n_h, 1))
        self._W2 = rng.standard_normal((n_y, self._n_h)) * numpy.sqrt(2/self._W1.shape[0])
        self._b2 = numpy.zeros((n_y, 1))        
        #print(f"W1: {self._W1.shape}, W2: {self._W2.shape}, b1: {self._b1.shape}, b2: {self._b2.shape}")
        # Visualize the data:
        plt.scatter(self._X[0, :], self._X[1, :], c=self._Y, s=40, cmap=plt.cm.Spectral)

    def _load_extra_datasets(self, N: int):
        #N = 200
        noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
        noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
        blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
        gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
        no_structure = rng.random((N, 2)), rng.random((N, 2))
        return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
    
    def _compute_cost(self, A2):
        """
        Computes the cross-entropy cost given in equation (13)
        
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost given equation (13)
        """
        m = self._Y.shape[1] # number of examples

        # Compute the cross-entropy cost
        cost1 = -(numpy.log(A2) @ self._Y.T)
        cost2 = -(numpy.log(1-A2) @ (1-self._Y).T)
        cost = (cost1 + cost2) / m
        #print(f"{cost1} -> {float(numpy.squeeze(cost1))}, cost2: {cost2}, cost3: {cost3} -> {float(numpy.squeeze(cost3))}")
        cost = float(numpy.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                        # E.g., turns [[17]] into 17  
        return cost
    
    def _forward_propagation(self, X = None):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)
        
        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        if X is None:
            X = self._X
        # Implement Forward Propagation to calculate A2 (probabilities)
        # (≈ 4 lines of code)
        Z1 = self._W1 @ X + self._b1
        A1 = numpy.tanh(Z1) # (n_h, m)
        Z2 = self._W2 @ A1 + self._b2
        A2 = sigmoid(Z2)
        assert(A2.shape == (1, X.shape[1]))
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        return A2, cache
    
    def _backward_propagation(self, cache):
        """
        Implement the backward propagation using the instructions above.
        
        Arguments:
        parameters -- python dictionary containing our parameters 
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = self._X.shape[1]
        #print(f"W1: {self._W1.shape}, W2: {self._W2.shape}")
        # Retrieve also A1 and A2 from dictionary "cache".
        #(≈ 2 lines of code)
        A1 = cache["A1"]
        A2 = cache["A2"]
        #print(f"A1: {A1.shape}, A2: {A2.shape}")
    
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        #(≈ 6 lines of code, corresponding to 6 equations on slide above)
        dZ2 = A2 - self._Y
        dW2 = (dZ2 @ A1.T) / m
        db2 = numpy.sum(dZ2, axis=1, keepdims=True) / m # Sum the columns of every row
        dg_dz1 = 1 - numpy.power(A1, 2)
        dZ1 = (self._W2.T @ dZ2) * dg_dz1
        dW1 = (dZ1 @ self._X.T) / m
        db1 = numpy.sum(dZ1, axis=1, keepdims=True) / m # Sum the columns of every row
        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        return grads

    def _update_parameters(self, grads):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients 
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """       
        # Retrieve each gradient from the dictionary "grads"
        #(≈ 4 lines of code)
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
        
        # Update rule for each parameter
        #(≈ 4 lines of code)
        self._W1 -= self._learning_rate * dW1
        self._b1 -= self._learning_rate * db1
        self._W2 -= self._learning_rate * dW2
        self._b2 -= self._learning_rate * db2

if __name__ == "__main__":
    classifier = PlanarDataClassification(1.2, 10000)
    classifier.BuildModel(True)
    classifier.Predict()
    classifier.plot_decision_boundary()
    plt.show()
    classifier.TuneHiddenLayerSizes([4,5,7,9], 5000)
    plt.show()