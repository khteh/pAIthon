import argparse, numpy, h5py
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.optimizers import Adam
from utils.GPU import InitializeGPU
from utils.TensorModelPlot import PlotModelHistory
class SmilingFaceConvNet():
    """
    A convolution NN which determines if the people in the input images are smiling.
    """
    _X_train: numpy.array = None
    _Y_train: numpy.array = None
    _X_test: numpy.array = None
    _Y_test: numpy.array = None
    _classes: numpy.array = None
    _model: tf.keras.Sequential = None
    _model_path: str = None
    _trained: bool = False
    def __init__(self, path:str):
        InitializeGPU()
        self._model_path = path
        self._prepare_data()
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._model = tf.keras.models.load_model(self._model_path)
            self._trained = True

    def _prepare_data(self):
        train_dataset = h5py.File('data/train_happy.h5', "r")
        self._X_train = numpy.array(train_dataset["train_set_x"][:]) # your train set features
        self._Y_train = numpy.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('data/test_happy.h5', "r")
        self._X_test = numpy.array(test_dataset["test_set_x"][:]) # your test set features
        self._Y_test = numpy.array(test_dataset["test_set_y"][:]) # your test set labels

        self._classes = numpy.array(test_dataset["list_classes"][:]) # the list of classes
        
        self._Y_train = self._Y_train.reshape((1, self._Y_train.shape[0]))
        self._Y_test = self._Y_test.reshape((1, self._Y_test.shape[0]))
        # Normalize image vectors
        self._X_train = self._X_train/255.
        self._X_test = self._X_test/255.

        # Reshape
        self._Y_train = self._Y_train.T
        self._Y_test = self._Y_test.T

        print ("number of training examples = " + str(self._X_train.shape[0]))
        print ("number of test examples = " + str(self._X_test.shape[0]))
        print ("X_train shape: " + str(self._X_train.shape))
        print ("Y_train shape: " + str(self._Y_train.shape))
        print ("X_test shape: " + str(self._X_test.shape))
        print ("Y_test shape: " + str(self._Y_test.shape))
        print(f"Class#: {self._classes}")

    def BuildModel(self, rebuild:bool = False, learning_rate: float = 0.01):
        """
        Implements the forward propagation for the binary classification model:
        ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
        
        Note that for simplicity and grading purposes, you'll hard-code all the values
        such as the stride and kernel (filter) sizes. 
        Normally, functions should take these values as function parameters.
        
        Arguments:
        None

        Returns:
        model -- TF Keras model (object containing the information for the entire training process) 
        """
        if self._model and not rebuild:
            return
        self._model = tf.keras.Sequential([
                ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
                layers.Input(shape=(64,64,3)),
                layers.ZeroPadding2D(padding=3),
                ## Conv2D with 32 7x7 filters and stride of 1
                layers.Conv2D(32, (7, 7), strides=(1, 1)),
                ## BatchNormalization for axis 3
                layers.BatchNormalization(axis=3), # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
                ## ReLU
                layers.ReLU(),
                ## Max Pooling 2D with default parameters
                layers.MaxPooling2D(),
                ## Flatten layer
                layers.Flatten(),
                ## Dense layer with 1 unit for output & 'sigmoid' activation
                layers.Dense(1)   # Linear activation ("pass-through") if not specified
            ])
        self._model.compile(
                loss=BinaryCrossentropy(from_logits=True), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to sigmoid activation which is typically used for binary classification
                optimizer=Adam(learning_rate=learning_rate), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                metrics=['accuracy']
            )
        self._model.summary()

    def TrainEvaluate(self, rebuild:bool, epochs:int, batch_size:int):
        if self._model:
            if not self._trained or rebuild:
                history = self._model.fit(self._X_train, self._Y_train, epochs=epochs, batch_size=batch_size)
                print(f"history: {history.history}")
                PlotModelHistory("Smiling face binary classifier", history)
                self._trained = True
                if self._model_path:
                    self._model.save(self._model_path)
                    print(f"Model saved to {self._model_path}.")
            self._model.evaluate(self._X_test, self._Y_test)
        else:
            raise RuntimeError("Please build the model first by calling BuildModel()!")

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Smiling face binary classifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    smilingFace = SmilingFaceConvNet("models/SmilingFaceClassifier.keras")
    smilingFace.BuildModel(args.retrain, 0.01)
    smilingFace.TrainEvaluate(args.retrain, 10, 16)