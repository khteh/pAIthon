import math, numpy
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.optimizers import Adam
from utils.TensorModelPlot import PlotModelHistory
from utils.GPU import InitializeGPU
class SignsLanguageDigits():
    _X_train: numpy.array = None
    _Y_train: numpy.array = None
    _X_test: numpy.array = None
    _Y_test: numpy.array = None

    _model: tf.keras.Sequential = None
    def __init__(self):
        InitializeGPU()
        self.PrepareData()

    def PrepareData(self):
        train_dataset = h5py.File('data/train_signs.h5', "r")
        self._X_train = numpy.array(train_dataset["train_set_x"][:]) # your train set features
        self._Y_train = numpy.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('data/test_signs.h5', "r")
        self._X_test = numpy.array(test_dataset["test_set_x"][:]) # your test set features
        self._Y_test = numpy.array(test_dataset["test_set_y"][:]) # your test set labels

        self._classes = numpy.array(test_dataset["list_classes"][:]) # the list of classes

        self._Y_train = self._Y_train.reshape((1, self._Y_train.shape[0]))
        self._Y_test = self._Y_test.reshape((1, self._Y_test.shape[0]))

        # Normalize image vectors
        self._X_train = self._X_train/255.
        self._X_test = self._X_test/255.

        # Reshape
        self._convert_labels_to_one_hot()

        print ("number of training examples = " + str(self._X_train.shape[0]))
        print ("number of test examples = " + str(self._X_test.shape[0]))
        print ("X_train shape: " + str(self._X_train.shape))
        print ("Y_train shape: " + str(self._Y_train.shape))
        print ("X_test shape: " + str(self._X_test.shape))
        print ("Y_test shape: " + str(self._Y_test.shape))
        print(f"Class#: {self._classes} {self._classes.shape}")
        print(f"Y_train: {self._Y_train[:10]}")
        print(f"Y_test: {self._Y_test[:10]}")

    def _convert_labels_to_one_hot(self):
        self._Y_train = numpy.eye(len(self._classes))[self._Y_train.reshape(-1)].T
        self._Y_test = numpy.eye(len(self._classes))[self._Y_test.reshape(-1)].T
        self._Y_train = self._Y_train.T
        self._Y_test = self._Y_test.T
        
    def BuildModel(self):
        """
        Implements the forward propagation for the model:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
        
        Note that for simplicity and grading purposes, you'll hard-code some values
        such as the stride and kernel (filter) sizes. 
        Normally, functions should take these values as function parameters.
        
        Arguments:
        input_img -- input dataset, of shape (input_shape)

        Returns:
        model -- TF Keras model (object containing the information for the entire training process) 
        """
        self._model = tf.keras.Sequential([
                layers.Input(shape=(64,64,3)),
                ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
                layers.Conv2D(8, (4,4), strides=(1,1), padding="same", name="L1"),
                ## ReLU
                layers.ReLU(name="L2"),
                ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
                layers.MaxPool2D((8,8), strides=(8,8), padding="same", name="L3"),
                ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
                layers.Conv2D(16, (2,2), strides=(1,1), padding="same", name="L4"),
                ## ReLU
                layers.ReLU(name="L5"),
                ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
                layers.MaxPool2D((4,4), strides=(4,4), padding="same", name="L6"),
                ## Flatten layer
                layers.Flatten(),
                ## Dense layer with 1 unit for output & 'sigmoid' activation
                layers.Dense(6)   # Linear activation ("pass-through") if not specified. Since the labels are 6 categories, use CategoricalCrossentropy
            ])
        self._model.compile(
                loss=CategoricalCrossentropy(from_logits=True), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
                optimizer=Adam(learning_rate=0.01), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                metrics=['accuracy']
            )
        self._model.summary()

    def TrainEvaluate(self, epochs:int, batch_size:int):
        train_dataset = tf.data.Dataset.from_tensor_slices((self._X_train, self._Y_train)).batch(64)
        validation_dataset = tf.data.Dataset.from_tensor_slices((self._X_test, self._Y_test)).batch(64)
        history = self._model.fit(train_dataset, epochs=100, validation_data=validation_dataset)
        print(f"history: {history.history}")
        PlotModelHistory("CNN", history)

if __name__ == "__main__":
    signs = SignsLanguageDigits()
    signs.BuildModel()
    signs.TrainEvaluate(100, 64)