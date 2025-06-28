# for array computations and loading data
import numpy
# for building and training neural networks
import tensorflow as tf
# custom functions
import utils
# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# reduce display precision on numpy arrays
numpy.set_printoptions(precision=2)

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

#TODO: Add similar classes for NN and Classification

class LinearRegressionModelEvaluationAndSelection():
    _X_train = None
    _X_train_scaled = None

    _X_cv = None
    _X_cv_scaled = None

    _X_test = None
    _X_test_scaled = None

    _Y_train = None
    _Y_cv = None
    _y_test = None

    _scaler: StandardScaler = None
    _model: LinearRegression = None

    _train_mse: float = None
    _cv_mse: float = None
    _test_mse: float = None

    _degree: int = None # Best performing linear regression model degree which yields the lowest mse.
    _poly : PolynomialFeatures = None

    def __init__(self, path):
        self.PrepareData(path)
        self.ScaleData(self._X_train)
        self.ScaleData(self._X_cv)
        self.ScaleData(self._X_test)

    def PrepareData(self, path: str):
        print(f"\n=== {self.PrepareData.__name__} ===")
        # Load the dataset from the text file
        data = numpy.loadtxt(path, delimiter=',') # './data/data_w3_ex1.csv'

        # Split the inputs and outputs into separate arrays
        x = data[:,0]
        y = data[:,1]

        print(f"the shape of the inputs x is: {x.shape}")
        print(f"the shape of the targets y is: {y.shape}")

        # Convert 1-D arrays into 2-D because the commands later will require it
        x = numpy.expand_dims(x, axis=1)
        y = numpy.expand_dims(y, axis=1)

        print(f"the shape of the inputs x is: {x.shape}")
        print(f"the shape of the targets y is: {y.shape}")
        # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
        self._X_train, x_, self._Y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

        # Split the 40% subset above into two: one half for cross validation and the other for the test set
        self._X_cv, self._X_test, self._Y_cv, self._Y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

        # Delete temporary variables
        del x_, y_

        print(f"the shape of the training set (input) is: {self._X_train.shape}")
        print(f"the shape of the training set (target) is: {self._Y_train.shape}\n")
        print(f"the shape of the cross validation set (input) is: {self._X_cv.shape}")
        print(f"the shape of the cross validation set (target) is: {self._Y_cv.shape}\n")
        print(f"the shape of the test set (input) is: {self._X_test.shape}")
        print(f"the shape of the test set (target) is: {self._Y_test.shape}")

        # Data set plot to show which points were used as training, cross validation, or test data.
        utils.plot_train_cv_test(self._X_train, self._Y_train, self._X_cv, self._Y_cv, self._X_test, self._Y_test, title="input vs. target")
        """
        StandardScaler from scikitlearn computes the z-score of your inputs. As a refresher, the z-score is given by the equation:
            z = (x - 𝜇) / lambda
        where  𝜇 is the mean of the feature values and lambda is the standard deviation. 
        """

    def ScaleData(self, data):
        # Compute the mean and standard deviation of the training set then transform it
        if not self._scaler:
            self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(data)
        print(f"Computed mean of the data: {self._scaler.mean_.squeeze():.2f}")
        print(f"Computed standard deviation of the data: {self._scaler.scale_.squeeze():.2f}")

        # Plot the results
        #utils.plot_dataset(x=data, y=y_train, title="scaled input vs. target")
        return scaled_data

    def RegressionModel(self, X, y):
        """
        Create a linear or polynomial regression model using the input data
        """
        # Initialize the class
        if not self._model:
            self._model = LinearRegression()

        # Train the model
        self._model.fit(X, y)

    def EvaluateLinearModel(self) -> float:
        """
        Args:
            
        To evaluate the performance of your model, you will measure the error for the training and cross validation sets. For the training error, recall the equation for calculating the mean squared error (MSE):
        J(W,b) = sum((f_w_b(X) - y) ** 2) / 2 * m
        Scikit-learn also has a built-in mean_squared_error() function that you can use. Take note though that as per the documentation, scikit-learn's implementation only divides by m and not 2*m, where m is the number of examples. As mentioned in Course 1 of this Specialization (cost function lectures), dividing by 2m is a convention we will follow but the calculations should still work whether or not you include it. Thus, to match the equation above, you can use the scikit-learn function then divide by 2 as shown below. We also included a for-loop implementation so you can check that it's equal.
        Another thing to take note: Since you trained the model on scaled values (i.e. using the z-score), you should also feed in the scaled training set instead of its raw values.    
        """
        # Firstly, evaluate model against test data set
        # Feed the scaled training set and get the predictions
        yhat = self._model.predict(self._X_train_scaled)

        # Use scikit-learn's utility function and divide by 2
        print(f"training MSE (using sklearn function): {mean_squared_error(self._Y_train, yhat) / 2}")

        # for-loop implementation
        total_squared_error = 0

        for i in range(len(yhat)):
            squared_error_i  = (yhat[i] - self._Y_train[i])**2
            total_squared_error += squared_error_i

        self._train_mse = total_squared_error / (2*len(yhat))
        print(f"training MSE (for-loop implementation): {self._train_mse.squeeze()}") # Remove axes of length one from mse.

        # Secondly, evaluate model against cross-validation data set
        """
        As with the training set, you will also want to scale the cross validation set. An important thing to note when using the z-score is you have to use the mean and standard deviation of the training set when scaling the cross validation set. This is to ensure that your input features are transformed as expected by the model. One way to gain intuition is with this scenario:

        Say that your training set has an input feature equal to 500 which is scaled down to 0.5 using the z-score.
        After training, your model is able to accurately map this scaled input x=0.5 to the target output y=300.
        Now let's say that you deployed this model and one of your users fed it a sample equal to 500.
        If you get this input sample's z-score using any other values of the mean and standard deviation, then it might not be scaled to 0.5 and your model will most likely make a wrong prediction (i.e. not equal to y=300).
        You will scale the cross validation set below by using the same StandardScaler you used earlier but only calling its transform() method instead of fit_transform().    
        """
        # Feed the scaled cross validation set
        yhat = self._model.predict(self._X_cv_scaled)

        # Use scikit-learn's utility function and divide by 2
        self._cv_mse = mean_squared_error(self._Y_cv, yhat) / 2
        print(f"Cross validation MSE: {self._cv_mse}")
        return self._cv_mse

    def AddPolynomialFeatures(self, degree: int = 2, include_bias: bool = False):
        """
        From the graphs earlier, you may have noticed that the target y rises more sharply at smaller values of x compared to higher ones. A straight line might not be the best choice because the target y seems to flatten out as x increases. 
        Now that you have these values of the training and cross validation MSE from the linear model, you can try adding polynomial features to see if you can get a better performance. The code will mostly be the same but with a few extra preprocessing steps.
        First, you will generate the polynomial features from your training set. The code below demonstrates how to do this using the PolynomialFeatures class. It will create a new input feature which has the squared values of the input x (i.e. degree=2).    
        """
        # Instantiate the class to make polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)

        # Compute the number of features and transform the training set
        X_train_mapped = poly.fit_transform(self._X_train)

        # Preview the first 5 elements of the new training set. Left column is `x` and right column is `x^2`
        # Note: The `e+<number>` in the output denotes how many places the decimal point should 
        # be moved. For example, `3.24e+03` is equal to `3240`
        print(X_train_mapped[:5])

        # Scale the inputs as before to narrow down the range of values.
        # Instantiate the class
        #self._scaler = StandardScaler()

        # Compute the mean and standard deviation of the training set then transform it
        #X_train_mapped_scaled = self._scaler.fit_transform(X_train_mapped)
        self._X_train_scaled = self.ScaleData(X_train_mapped)

        # Preview the first 5 elements of the scaled training set.
        print(self._X_train_scaled[:5])

        # Add the polynomial features to the cross validation set
        X_cv_mapped = poly.transform(self._X_cv)

        # Scale the cross validation set using the mean and standard deviation of the training set
        #X_cv_mapped_scaled = standard_scaler.transform(X_cv_mapped)
        self._X_cv_scaled = self.ScaleData(X_cv_mapped)

    def ModelSelection(self, max_degree: int):
        # Initialize lists to save the errors, models, and feature transforms
        train_mses = []
        cv_mses = []
        models = []
        polys = []
        scalers = []

        # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
        for degree in range(1,max_degree + 1):
            
            # Add polynomial features to the training set
            poly = PolynomialFeatures(degree, include_bias=False)
            X_train_mapped = poly.fit_transform(self._X_train)
            polys.append(poly)
            
            # Scale the training set
            standard_scaler = StandardScaler()
            X_train_mapped_scaled = standard_scaler.fit_transform(X_train_mapped)
            scalers.append(standard_scaler)

            # Add polynomial features and scale the cross validation set
            X_cv_mapped = poly.transform(self._X_cv)
            X_cv_mapped_scaled = standard_scaler.transform(X_cv_mapped)

            # Create and train the model
            model = LinearRegression()
            models.append(model)
            
            # Compute the training MSE
            yhat = model.predict(X_train_mapped_scaled)
            train_mses.append(mean_squared_error(self._Y_train, yhat) / 2)
        
            # Compute the cross validation MSE
            yhat = model.predict(X_cv_mapped_scaled)
            cv_mses.append(mean_squared_error(self._Y_cv, yhat) / 2)
            
        # Plot the results
        degrees=range(1,11)
        utils.plot_train_cv_mses(degrees, train_mses, cv_mses, title="degree of polynomial vs. train and CV MSEs")
        # use the model with the lowest cv_mse as the one best suited for your application.
        # Get the model with the lowest CV MSE (add 1 because list indices start at 0)
        # This also corresponds to the degree of the polynomial added
        self._degree = numpy.argmin(cv_mses) + 1
        self._poly = polys[self._degree - 1]
        self._scaler = scalers[self._degree - 1]
        self._model = models[self._degree - 1]
        self._train_mse = train_mses[self._degree - 1]
        self._cv_mse = cv_mses[self._degree - 1]
        print(f"Lowest CV MSE is found in the model with degree={self._degree}")

    def TestDataSetPerformance(self):
        """
        Obtain and publish the generalization error by computing the test set's MSE. As usual, you should transform this data the same way you did with the training and cross validation sets.
        """
        # Add polynomial features to the test set
        X_test_mapped = self._poly.transform(self._X_test)

        # Scale the test set
        X_test_mapped_scaled = self._scaler.transform(X_test_mapped)

        # Compute the test MSE
        yhat = self._model.predict(X_test_mapped_scaled)
        self._test_mse = mean_squared_error(self._Y_test, yhat) / 2

        print(f"Training MSE: {self._train_mse:.2f}")
        print(f"Cross Validation MSE: {self._cv_mse:.2f}")
        print(f"Test MSE: {self._test_mse:.2f}")

if __name__ == "__main__":
    model = LinearRegressionModelEvaluationAndSelection('./data/data_w3_ex1.csv')
    linear_mse = model.EvaluateLinearModel()
    model.AddPolynomialFeatures()
    poly_mse = model.EvaluateLinearModel()
    if linear_mse <= poly_mse:
        print("Linear model is the right choice")
    else:
        print("Polynomial model is the right choice")
    model.ModelSelection(10)
    model.TestDataSetPerformance()