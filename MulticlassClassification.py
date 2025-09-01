import logging, numpy
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from utils.lab_utils_multiclass_TF import *

numpy.set_printoptions(precision=2)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
"""
A network of this type will have multiple units in its final layer. Each output is associated with a category. 
When an input example is applied to the network, the output with the highest value is the category predicted. 
If the output is applied to a softmax function, the output of the softmax will provide probabilities of the input being in each category.
"""
class MulticlassClassification():
    _X_train = None
    _Y_train = None
    def __init__(self):
        self.PrepareData()

    def PrepareData(self):
        # make 4-class dataset for classification
        classes = 4
        m = 100
        coordinates = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
        std = 1.0
        self._X_train, self._Y_train = make_blobs(n_samples=m, centers=coordinates, cluster_std=std,random_state=30)
        plt_mc(self._X_train,self._Y_train,classes, coordinates, std=std)
        # show classes in data set
        print(f"unique classes {numpy.unique(self._Y_train)}")
        # show how classes are represented
        print(f"class representation {self._Y_train[:10]}")
        # show shapes of our dataset
        print(f"shape of self._X_train: {self._X_train.shape}, shape of self._Y_train: {self._Y_train.shape}")    
        return self._X_train, self._Y_train

    def MulticlassClassification(self):
        """
        This network has four outputs, one for each class. Given an input example, the output with the highest value is the predicted class of the input.
        Below is an example of how to construct this network in Tensorflow. Notice the output layer uses a linear rather than a softmax activation. 
        While it is possible to include the softmax in the output layer, it is more numerically stable if linear outputs are passed to the loss function during training. 
        If the model is used to predict probabilities, the softmax can be applied at that point.
        L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection by "turning off" less important features or nodes in the network.
        L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero. This helps to prevent individual weights from becoming excessively large and dominating the model.
        """
        classes = 4
        tf.random.set_seed(1234)  # applied to achieve consistent results
        model = Sequential(
            [
                Dense(2, activation = 'relu',   name = "L1", kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected
                Dense(4, activation = 'linear', name = "L2")  # Linear activation ("pass-through") if not specified
            ]
        )
        # Setting from_logits=True as an argument to the loss function specifies that the output activation was linear rather than a softmax.
        model.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(0.01), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
        )
        model.fit(
            self._X_train,self._Y_train,
            epochs=200
        )
        plt_cat_mc(self._X_train, self._Y_train, model, classes)

if __name__ == "__main__":
    multiclassClassification = MulticlassClassification()
    multiclassClassification.MulticlassClassification()