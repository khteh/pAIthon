import numpy, copy, matplotlib.pyplot as plt, h5py, scipy
from Activations import sigmoid, relu, relu_backward, sigmoid_backward
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

class LogisticRegressionDeepNN():
    _train_path:str = None
    _test_path:str = None
    _X_train_orig = None
    _X_test_orig = None
    _classes = None
    _X_train = None
    _Y_train = None
    _X_test = None
    _Y_test = None
    _layers_dims = None
    _epochs:int = None
    _learning_rate: float = None
    _parameters = None
    _costs = None

    def __init__(self, trainpath:str, testpath:str, learning_rate:float, epochs:int):
        self._train_path = trainpath
        self._test_path = testpath
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
        self._PrepareData()

    def _PrepareData(self):
        train_dataset = h5py.File(self._train_path, "r")
        self._X_train_orig = numpy.array(train_dataset["train_set_x"][:]) # your train set features
        self._Y_train = numpy.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File(self._test_path, "r")
        self._X_test_orig = numpy.array(test_dataset["test_set_x"][:]) # your test set features
        self._Y_test = numpy.array(test_dataset["test_set_y"][:]) # your test set labels

        self._classes = numpy.array(test_dataset["list_classes"][:]) # the list of classes
        
        self._Y_train = self._Y_train.reshape((1, self._Y_train.shape[0]))
        self._Y_test = self._Y_test.reshape((1, self._Y_test.shape[0]))

        self._X_train = self._X_train_orig.reshape(self._X_train_orig.shape[0], -1).T / 255
        self._X_test = self._X_test_orig.reshape(self._X_test_orig.shape[0], -1).T / 255
        print(f"X_train: {self._X_train.shape}, Y_train: {self._Y_train.shape}, X_test: {self._X_test.shape}, Y_test: {self._Y_test.shape}")

    def BuildTrainModel(self, print_cost=False):
        L = len(self._layers_dims) # number of layers in the network
        parameters = {}
        costs = []
        for l in range(1, L):
            parameters['W' + str(l)] = rng.standard_normal((self._layers_dims[l], self._layers_dims[l-1])) * numpy.sqrt(2/self._layers_dims[l-1]) # ReLU
            parameters['b' + str(l)] = numpy.zeros((self._layers_dims[l], 1))            
            assert(parameters['W' + str(l)].shape == (self._layers_dims[l], self._layers_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (self._layers_dims[l], 1))
        # Loop (gradient descent)
        for i in range(0, self._epochs):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self._L_model_forward(self._X_train, parameters)

            # Compute cost.
            cost = self._compute_cost(AL, self._Y_train)
        
            # Backward propagation.
            grads = self._L_model_backward(AL, self._Y_train, caches)
    
            # Update parameters.
            parameters = self._update_parameters(parameters, grads)
                    
            # Print the cost every 100 iterations and for the last iteration
            if print_cost and (i % 100 == 0 or i == self._epochs - 1):
                print("Cost after iteration {}: {}".format(i, numpy.squeeze(cost)))
            if i % 100 == 0:
                costs.append(cost)
        self._parameters = parameters
        self._costs = costs
        self._plot_costs()

    def Predict(self):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        m = self._X_train.shape[1]
        n = len(self._parameters) // 2 # number of layers in the neural network
        p_train = numpy.zeros((1,m))
        
        # Forward propagation
        probas, caches = self._L_model_forward(self._X_train, self._parameters)
        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p_train[0,i] = 1
            else:
                p_train[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Train Accuracy: "  + str(numpy.sum((p_train == self._Y_train)/m)))

        m = self._X_test.shape[1]
        n = len(self._parameters) // 2 # number of layers in the neural network
        p_test = numpy.zeros((1,m))
        
        # Forward propagation
        probas, caches = self._L_model_forward(self._X_test, self._parameters)
        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p_test[0,i] = 1
            else:
                p_test[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Test Accuracy: "  + str(numpy.sum((p_test == self._Y_test)/m)))
        self._print_mislabeled_images(p_test)
        return p_train, p_test

    def _print_mislabeled_images(self, p):
        """
        Plots images where predictions and truth were different.
        X -- dataset
        y -- true labels
        p -- predictions
        """
        print(f"\n=== {self._print_mislabeled_images.__name__} ===")
        a = p + self._Y_test
        mislabeled_indices = numpy.asarray(numpy.where(a == 1))
        plt.figure(figsize=(40, 10), constrained_layout=True)
        #fig.tight_layout(pad=5, rect=[0, 0.03, 1, 0.95]) #[left, bottom, right, top]
        num_images = len(mislabeled_indices[0])
        for i in range(num_images):
            index = mislabeled_indices[1][i]
            plt.subplot(2, num_images, i + 1)
            plt.imshow(self._X_test[:,index].reshape(64,64,3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + self._classes[int(p[0,index])].decode("utf-8") + " \n Class: " + self._classes[self._Y_test[0,index]].decode("utf-8"), fontsize=18)
        plt.show()
        
    def _L_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """
        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        #print(f"L: {L}, params: {parameters}")
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        # The for loop starts at 1 because layer 0 is the input
        for l in range(1, L):
            A_prev = A 
            A, cache = self._linear_activation_forward(A_prev, parameters[f"W{l}"], parameters[f"b{l}"], "relu")
            caches.append(cache)
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        #(≈ 2 lines of code)
        AL, cache = self._linear_activation_forward(A, parameters[f"W{L}"], parameters[f"b{L}"], "sigmoid")
        caches.append(cache)            
        return AL, caches

    def _L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = - (numpy.divide(Y,AL) - numpy.divide(1-Y, 1-AL))
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[-1]
        dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(dAL, current_cache, "sigmoid")
        grads[f"dA{L-1}"] = dA_prev_temp
        grads[f"dW{L}"] = dW_temp
        grads[f"db{L}"] = db_temp
        #print(f"L: {L}, grads: {grads}")
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            #print(f"Layer {l}...")
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(dA_prev_temp, current_cache, "relu")
            grads[f"dA{l}"] = dA_prev_temp
            grads[f"dW{l+1}"] = dW_temp
            grads[f"db{l+1}"] = db_temp
        return grads
    
    def _compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]

        # Compute loss from aL and y.
        cost1 = -(numpy.log(AL) @ Y.T)
        cost2 = -(numpy.log(1-AL) @ (1-Y).T)
        cost = (cost1 + cost2) / m
        cost = numpy.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        return cost

    def _plot_costs(self):
        plt.figure(figsize=(10, 10), constrained_layout=True)
        plt.plot(numpy.squeeze(self._costs))
        plt.ylabel('cost', fontsize=20)
        plt.xlabel('iterations (per hundreds)', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(f"Learning rate = {self._learning_rate}", fontsize=22, fontweight="bold")
        plt.show()

    def _linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        Z = W @ A + b
        cache = (A, W, b)
        return Z, cache
    
    def _linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        """
        if activation == "sigmoid":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A = sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A = relu(Z)
        cache = (linear_cache, Z)
        return A, cache

    def _linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (dZ @ A_prev.T) / m
        db = numpy.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = W.T @ dZ   
        return dA_prev, dW, db
    
    def _linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def _update_parameters(self, params, grads):
        """
        Update parameters using gradient descent
        
        Arguments:
        params -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                    parameters["W" + str(l)] = ... 
                    parameters["b" + str(l)] = ...
        """
        parameters = copy.deepcopy(params)
        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] -= self._learning_rate * grads[f"dW{l+1}"]
            parameters["b" + str(l+1)] -= self._learning_rate * grads[f"db{l+1}"]
        return parameters
    
if __name__ == "__main__":
    classifier = LogisticRegressionDeepNN("data/train_catvnoncat.h5", "data/test_catvnoncat.h5", 0.001, 3000)
    classifier.BuildTrainModel(True)
    p_train, p_test = classifier.Predict()