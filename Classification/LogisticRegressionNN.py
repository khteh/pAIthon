import numpy, copy, matplotlib.pyplot as plt, h5py, scipy
from PIL import Image
from scipy import ndimage
from Activations import sigmoid
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

class LogisticRegressionNN():
    _train_path:str = None
    _test_path:str = None
    _X_train_orig = None
    _X_test_orig = None
    _classes = None
    _X_train = None
    _Y_train = None
    _X_test = None
    _Y_test = None
    _epochs:int = None
    _learning_rate: float = None
    def __init__(self, trainpath:str, testpath:str, learning_rate:float, epochs:int):
        self._train_path = trainpath
        self._test_path = testpath
        self._learning_rate = learning_rate
        self._epochs = epochs
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

    def _propagate(self, w, b):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        grads -- dictionary containing the gradients of the weights and bias
                (dw -- gradient of the loss with respect to w, thus same shape as w)
                (db -- gradient of the loss with respect to b, thus same shape as b)
        cost -- negative log-likelihood cost for logistic regression
        
        Tips:
        - Write your code step by step for the propagation. numpy.log(), numpy.dot()
        """
        m = self._X_train.shape[1]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        #(≈ 2 lines of code)
        # compute activation
        # A = ...
        # compute cost by using numpy.dot to perform multiplication. 
        # And don't use loops for the sum.
        # cost = ...                                
        predictions = sigmoid(w.T @ self._X_train + b) # (1, m)
        assert (1,m) == predictions.shape
        cost = numpy.sum(-self._Y_train * numpy.log(predictions) - (1 - self._Y_train) * numpy.log(1 - predictions)) / m # scalar

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = ((predictions - self._Y_train) @ self._X_train.T) / m # (1,m) @ (m,n) + (1,n) = (1,n)
        db = numpy.sum(predictions - self._Y_train) / m # scalar
        #print(f"dw: {dw.shape}")
        #assert dw.shape == w.shape
        cost = numpy.squeeze(numpy.array(cost))       
        grads = {"dw": dw.T,
                "db": db}
        return grads, cost
    
    def _optimize(self, w, b, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        
        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """
        w = copy.deepcopy(w)
        b = copy.deepcopy(b)
        costs = []
        for i in range(self._epochs):
            # (≈ 1 lines of code)
            # Cost and gradient calculation 
            # grads, cost = ...
            grads, cost = self._propagate(w,b)
            
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            # update rule (≈ 2 lines of code)
            # w = ...
            # b = ...
            w = w - self._learning_rate * dw
            b = b - self._learning_rate * db
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
                # Print the cost every 100 training iterations
                if print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        params = {"w": w,
                "b": b}
        grads = {"dw": dw,
                "db": db}
        return params, grads, costs

    def BuildTrainModel(self, print_cost=False):
        """
        Builds the logistic regression model by calling the function you've implemented previously
        
        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to True to print the cost every 100 iterations
        
        Returns:
        d -- dictionary containing information about the model.
        """
        # (≈ 1 line of code)   
        # initialize parameters with zeros
        # and use the "shape" function to get the first dimension of X_train
        # w, b = ...
        w = rng.standard_normal((self._X_train.shape[0], 1)) * numpy.sqrt(2/self._X_train.shape[1]) # ReLU
        b = 0.0
        #(≈ 1 line of code)
        # Gradient descent 
        """
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)   
        """
        print(f"w: {w.shape}, b: {b}, X_train: {self._X_train.shape}, Y_train: {self._Y_train.shape}")
        params, grads, costs = self._optimize(w, b, print_cost)
        
        # Retrieve parameters w and b from dictionary "params"
        w = params["w"]
        b = params["b"]
        
        # Predict test/train set examples (≈ 2 lines of code)
        Y_prediction_test = self.predict(w, b, self._X_test)
        Y_prediction_train = self.predict(w, b, self._X_train)
        
        # Print train/test Errors
        if print_cost:
            print("train accuracy: {} %".format(100 - numpy.mean(numpy.abs(Y_prediction_train - self._Y_train)) * 100))
            print("test accuracy: {} %".format(100 - numpy.mean(numpy.abs(Y_prediction_test - self._Y_test)) * 100))

        # Plot learning curve (with costs)
        costs = numpy.squeeze(costs)
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title(f"Learning rate = {self._learning_rate}")
        plt.show()
        d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : self._learning_rate,
            "num_iterations": self._epochs}
        return d
    
    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        m = X.shape[1]
        Y_prediction = numpy.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        #(≈ 1 line of code)
        # A = ...
        A = sigmoid(w.T @ X + b)
        
        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            #(≈ 4 lines of code)
            Y_prediction[0,i] = 1 if A[0, i] > 0.5 else 0
        return Y_prediction
    
if __name__ == "__main__":
    classifier = LogisticRegressionNN("data/train_catvnoncat.h5", "data/test_catvnoncat.h5", 0.001, 2000)
    classifier.BuildTrainModel(True)