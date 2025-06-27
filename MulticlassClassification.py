import logging, numpy
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
numpy.set_printoptions(precision=2)
from lab_utils_multiclass_TF import *
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
"""
A network of this type will have multiple units in its final layer. Each output is associated with a category. 
When an input example is applied to the network, the output with the highest value is the category predicted. 
If the output is applied to a softmax function, the output of the softmax will provide probabilities of the input being in each category.
"""
def PrepareData():
    # make 4-class dataset for classification
    classes = 4
    m = 100
    coordinates = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
    std = 1.0
    X_train, y_train = make_blobs(n_samples=m, centers=coordinates, cluster_std=std,random_state=30)
    plt_mc(X_train,y_train,classes, coordinates, std=std)
    # show classes in data set
    print(f"unique classes {numpy.unique(y_train)}")
    # show how classes are represented
    print(f"class representation {y_train[:10]}")
    # show shapes of our dataset
    print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")    
    return X_train, y_train

def MulticlassClassification(X_train, y_train):
    """
    This network has four outputs, one for each class. Given an input example, the output with the highest value is the predicted class of the input.
    Below is an example of how to construct this network in Tensorflow. Notice the output layer uses a linear rather than a softmax activation. 
    While it is possible to include the softmax in the output layer, it is more numerically stable if linear outputs are passed to the loss function during training. 
    If the model is used to predict probabilities, the softmax can be applied at that point.    
    """
    classes = 4
    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            Dense(2, activation = 'relu',   name = "L1"),
            Dense(4, activation = 'linear', name = "L2")
        ]
    )
    # Setting from_logits=True as an argument to the loss function specifies that the output activation was linear rather than a softmax.
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
    )
    model.fit(
        X_train,y_train,
        epochs=200
    )
    plt_cat_mc(X_train, y_train, model, classes)

if __name__ == "__main__":
    X_train, y_train = PrepareData()
    MulticlassClassification(X_train, y_train)