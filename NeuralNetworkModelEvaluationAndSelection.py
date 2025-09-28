# for array computations and loading data
import numpy, tensorflow as tf
# for building and training neural networks
# custom functions
# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from utils.GPU import InitializeGPU
from utils.Plots import *
# reduce display precision on numpy arrays
numpy.set_printoptions(precision=2)

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

class NeuralNetworkModelEvaluationAndSelection():
    """
    TODO: Cross-validation MSE much higher than training MSE.
    RESULTS:
    Model 1: Training MSE: 73.51, CV MSE: 2754.62
    Model 2: Training MSE: 406.08, CV MSE: 3696.82
    Model 3: Training MSE: 408.54, CV MSE: 3720.65
    the model with the lowest CV MSE is 0
    1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
    Training MSE: 73.51
    Cross Validation MSE: 2754.62
    Test MSE: 4314.66
    """
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
    _model: Sequential = None

    _train_mse: float = None
    _cv_mse: float = None
    _test_mse: float = None

    _degree: int = None # Best performing linear regression model degree which yields the lowest mse.
    _poly : PolynomialFeatures = None
    _models: list[Sequential] = None

    def __init__(self, path):
        InitializeGPU()
        self._prepare_data(path)

    def _prepare_data(self, path: str):
        print(f"\n=== {self._prepare_data.__name__} ===")
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
        plot_train_cv_test(self._X_train, self._Y_train, self._X_cv, self._Y_cv, self._X_test, self._Y_test, title="input vs. target")
        """
        StandardScaler from scikitlearn computes the z-score of your inputs. As a refresher, the z-score is given by the equation:
            z = (x - 𝜇) / lambda
        where  𝜇 is the mean of the feature values and lambda is the standard deviation. 
        """

    def ScaleData(self, data):
        print(f"\n=== {self.ScaleData.__name__} ===")
        # Compute the mean and standard deviation of the training set then transform it
        if not self._scaler:
            self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(data)
        print(f"Computed mean of the data: {self._scaler.mean_.squeeze():.2f}")
        print(f"Computed standard deviation of the data: {self._scaler.scale_.squeeze():.2f}")

        # Plot the results
        print(f"data: {data.shape}, Y_train: {self._Y_train.shape}")
        print(f"x: {data.shape}, y: {self._Y_train.shape}")
        if data.shape == self._Y_train.shape:
            plot_dataset(x=data, y=self._Y_train, title="scaled input vs. target")
        return scaled_data
    
    def BuildModels(self):
        tf.random.set_seed(20)
        model_1 = Sequential(
            [
                Dense(25, activation = 'relu', kernel_regularizer=l2(0.1)), # Densely connected, or fully connected
                Dense(15, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(1, activation = 'linear')
            ],
            name='model_1'
        )

        model_2 = Sequential(
            [
                Dense(20, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(12, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(12, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(20, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(1, activation = 'linear')
            ],
            name='model_2'
        )

        model_3 = Sequential(
            [
                Dense(32, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(16, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(8, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(4, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(12, activation = 'relu', kernel_regularizer=l2(0.1)),
                Dense(1, activation = 'linear')
            ],
            name='model_3'
        )
        self._models = [model_1, model_2, model_3]

    def AddPolynomialFeatures(self, degree: int = 1, bias: bool = False):
        """
        This is optional because neural networks can learn non-linear relationships so you can opt to skip adding polynomial features. 
        The code is still included below in case you want to try later and see what effect it will have on your results. 
        The default degree is set to 1 to indicate that it will just use x_train, x_cv, and x_test as is (i.e. without any additional polynomial features).
        """
        print(f"\n=== {self.AddPolynomialFeatures.__name__} ===")
        # Add polynomial features
        self._poly = PolynomialFeatures(degree, include_bias = bias)
        X_train_mapped = self._poly.fit_transform(self._X_train)
        X_cv_mapped = self._poly.transform(self._X_cv)
        X_test_mapped = self._poly.transform(self._X_test)
        self._X_train_scaled = self.ScaleData(X_train_mapped)
        self._X_cv_scaled = self.ScaleData(X_cv_mapped)
        self._X_test_scaled = self.ScaleData(X_test_mapped)

    def ModelSelection(self, rate: float):
        # Initialize lists that will contain the errors for each model
        nn_train_mses = []
        nn_cv_mses = []

        # Build the models
        self.BuildModels()

        # Loop over the the models
        for model in self._models:
            
            # Setup the loss and optimizer
            model.compile(
                loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=rate), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
            )

            print(f"Training {model.name}...")
            
            # Train the model
            model.fit(
                self._X_train_scaled, self._Y_train,
                epochs=300,
                verbose=0
            )
            print("Done!\n")
            
            # Record the training MSEs
            yhat = model.predict(self._X_train_scaled)
            train_mse = mean_squared_error(self._Y_train, yhat) / 2
            nn_train_mses.append(train_mse)
            
            # Record the cross validation MSEs 
            yhat = model.predict(self._X_cv_scaled)
            cv_mse = mean_squared_error(self._Y_cv, yhat) / 2
            nn_cv_mses.append(cv_mse)
            
        # print results
        print("RESULTS:")
        for model_num in range(len(nn_train_mses)):
            print(
                f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
                f"CV MSE: {nn_cv_mses[model_num]:.2f}"
                )
        # Select the model with the lowest CV MSE
        model = numpy.argmin(nn_cv_mses)
        print(f"The model with the lowest CV MSE is {model}")
        self._model = self._models[model]
        self._train_mse = nn_train_mses[model]
        self._cv_mse = nn_cv_mses[model]

    def TestDataSetPerformance(self):
        """
        Obtain and publish the generalization error by computing the test set's MSE. As usual, you should transform this data the same way you did with the training and cross validation sets.
        """
        # Compute the test MSE
        yhat = self._model.predict(self._X_test_scaled)
        self._test_mse = mean_squared_error(self._Y_test, yhat) / 2
        print(f"Training MSE: {self._train_mse:.2f}")
        print(f"Cross Validation MSE: {self._cv_mse:.2f}")
        print(f"Test MSE: {self._test_mse:.2f}")

if __name__ == "__main__":
    nn = NeuralNetworkModelEvaluationAndSelection('./data/data_w3_ex1.csv')
    nn.AddPolynomialFeatures()
    nn.ModelSelection(0.1)
    nn.TestDataSetPerformance()