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
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import layers, losses, optimizers, regularizers

# reduce display precision on numpy arrays
numpy.set_printoptions(precision=2)

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

class ClassificationModelEvaluationAndSelection():
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

    _train_error: float = None
    _cv_error: float = None
    _test_mse: float = None

    _degree: int = None # Best performing linear regression model degree which yields the lowest mse.
    _poly : PolynomialFeatures = None
    _models: list[Sequential] = None

    def __init__(self, path):
        self.PrepareData(path)
        self._X_train_scaled = self.ScaleData(self._X_train)
        self._X_cv_scaled = self.ScaleData(self._X_cv)
        self._X_test_scaled = self.ScaleData(self._X_test)

    def PrepareData(self, path: str):
        print(f"\n=== {self.PrepareData.__name__} ===")
        # Load the dataset from a text file
        data = numpy.loadtxt(path, delimiter=',')

        # Split the inputs and outputs into separate arrays
        x_bc = data[:,:-1]
        y_bc = data[:,-1]

        # Convert y into 2-D because the commands later will require it (x is already 2-D)
        y_bc = numpy.expand_dims(y_bc, axis=1)

        print(f"the shape of the inputs x is: {x_bc.shape}")
        print(f"the shape of the targets y is: {y_bc.shape}")

        # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
        self._X_train, x_, self._Y_train, y_ = train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)

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

    def BuildModels(self):
        tf.random.set_seed(20)
        model_1 = Sequential(
            [
                Dense(25, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(15, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(1, activation = 'linear')
            ],
            name='model_1'
        )

        model_2 = Sequential(
            [
                Dense(20, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(12, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(12, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(20, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(1, activation = 'linear')
            ],
            name='model_2'
        )

        model_3 = Sequential(
            [
                Dense(32, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(16, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(8, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(4, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(12, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
                Dense(1, activation = 'linear')
            ],
            name='model_3'
        )
        self._models = [model_1, model_2, model_3]

    def ModelSelection(self, threshold: float = 0.5):
        """
        To predict 1 only if very confident, use high value of threshold. This results in high precision, low recall
        To predict 1 even when in doubt, use low value of threshold. This results in low precision, high recall
        """
        # Initialize lists that will contain the errors for each model
        nn_train_error = []
        nn_cv_error = []

        # Build the models
        self.BuildModels()

        # Loop over each model
        for model in self._models:
            
            # Setup the loss and optimizer
            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            )
            print(f"Training {model.name}...")

            # Train the model
            model.fit(
                self._X_train_scaled, self._Y_train,
                epochs=200,
                verbose=0
            )
            print("Done!\n")
            
            # Record the fraction of misclassified examples for the training set
            yhat = model.predict(self._X_train_scaled)
            yhat = tf.math.sigmoid(yhat)
            yhat = numpy.where(yhat >= threshold, 1, 0)
            train_error = numpy.mean(yhat != self._Y_train)
            nn_train_error.append(train_error)

            # Record the fraction of misclassified examples for the cross validation set
            yhat = model.predict(self._X_cv_scaled)
            yhat = tf.math.sigmoid(yhat)
            yhat = numpy.where(yhat >= threshold, 1, 0)
            cv_error = numpy.mean(yhat != self._Y_cv)
            nn_cv_error.append(cv_error)

        # Print the result
        for model_num in range(len(nn_train_error)):
            print(
                f"Model {model_num+1}: Training Set Classification Error: {nn_train_error[model_num]:.5f}, " +
                f"CV Set Classification Error: {nn_cv_error[model_num]:.5f}"
                )
        # Select the model with the lowest CV MSE
        self._model = self._models[numpy.argmin(nn_cv_error)]
        self._train_error = nn_train_error[numpy.argmin(nn_cv_error)]
        self._cv_error = nn_cv_error[numpy.argmin(nn_cv_error)]

    def TestDataSetPerformance(self, threshold: float = 0.5):
        """
        Obtain and publish the generalization error by computing the test set's MSE. As usual, you should transform this data the same way you did with the training and cross validation sets.
        """
        # Compute the test MSE
        yhat = self._model.predict(self._X_test_scaled)
        yhat = tf.math.sigmoid(yhat)
        yhat = numpy.where(yhat >= threshold, 1, 0)
        nn_test_error = numpy.mean(yhat != self._Y_test)
        print(f"Training Set Classification Error: {self._train_error:.4f}")
        print(f"CV Set Classification Error: {self._cv_error:.4f}")
        print(f"Test Set Classification Error: {nn_test_error:.4f}")
if __name__ == "__main__":
    model = ClassificationModelEvaluationAndSelection('./data/data_w3_ex2.csv')
    model.ModelSelection(0.5)
    model.TestDataSetPerformance(0.5)