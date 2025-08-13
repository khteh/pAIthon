import numpy, logging, warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from utils.GPU import InitializeGPU
from autils import *
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.autograph.set_verbosity(0)

class HandWrittenDigitsNN():
    _X = None
    _Y = None
    _model: Sequential = None
    def __init__(self):
        InitializeGPU()
        self.PrepareData()
        self.VisualizeData()

    def PrepareData(self):
        """
        - The data set contains 1000 training examples of handwritten digits $^1$, here limited to zero and one.  

            - Each training example is a 20-pixel x 20-pixel grayscale image of the digit. 
                - Each pixel is represented by a floating-point number indicating the grayscale intensity at that location. 
                - The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. 
                - Each training example becomes a single row in our data matrix `X`. 
                - This gives us a 1000 x 400 matrix `X` where every row is a training example of a handwritten digit image.
        - The second part of the training set is a 1000 x 1 dimensional vector `y` that contains labels for the training set
            - `y = 0` if the image is of the digit `0`, `y = 1` if the image is of the digit `1`.

        This is a subset of the MNIST handwritten digit dataset (http://yann.lecun.com/exdb/mnist/)</sub>
        """
        self._X = numpy.load("data/X.npy")
        self._y = numpy.load("data/y.npy")
        self._X = self._X[0:1000]
        self._Y = self._Y[0:1000]

    def VisualizeData(self):
        m, n = self._X.shape
        fig, axes = plt.subplots(8,8, figsize=(8,8)) # figsize = (width, height)
        fig.tight_layout(pad=0.1)
        for i,ax in enumerate(axes.flat):
            # Select random indices
            random_index = numpy.random.randint(m)
            
            # Select rows corresponding to the random indices and
            # reshape the image
            X_random_reshaped = self._X[random_index].reshape((20,20)).T
            
            # Display the image
            ax.imshow(X_random_reshaped, cmap='gray')
            
            # Display the label above the image
            ax.set_title(self._Y[random_index,0])
            ax.set_axis_off()
        fig.suptitle("Label", fontsize=16)
        plt.show()

    def BuildModel(self):
        self._model = Sequential(
            [               
                tf.keras.Input(shape=(400,)),    #specify input size
                ### START CODE HERE ### 
                Dense(25, input_dim=400, activation='sigmoid', name="L1"),
                Dense(15, input_dim=25, activation='sigmoid', name="L2"),
                Dense(1, input_dim=15, activation='sigmoid', name="L3"),
                ### END CODE HERE ### 
            ], name = "my_model" 
        )
        self._model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.001),
        )
        self._model.fit(
            self._X, self._Y, epochs=20
        )        
        self._model.summary()

    def Predict(self):
        """
        The output of the model is interpreted as a probability. In the first example above, the input is a zero. The model predicts the probability that the input is a one is nearly zero. 
        In the second example, the input is a one. The model predicts the probability that the input is a one is nearly one.
        As in the case of logistic regression, the probability is compared to a threshold to make a final prediction.
        """
        prediction = self._model.predict(self._X[0].reshape(1,400))  # a zero
        print(f" predicting a zero: {prediction}")
        prediction = self._model.predict(self._X[500].reshape(1,400))  # a one
        print(f" predicting a one:  {prediction}")
        # The following code compares the predictions vs the labels for a random sample of 64 digits. This takes a moment to run.
        m, n = self._X.shape

        fig, axes = plt.subplots(8,8, figsize=(8,8)) # figsize = (width, height)
        fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

        for i,ax in enumerate(axes.flat):
            # Select random indices
            random_index = numpy.random.randint(m)
            
            # Select rows corresponding to the random indices and
            # reshape the image
            X_random_reshaped = self._X[random_index].reshape((20,20)).T
            
            # Display the image
            ax.imshow(X_random_reshaped, cmap='gray')
            
            # Predict using the Neural Network
            prediction = self._model.predict(self._X[random_index].reshape(1,400))
            if prediction >= 0.5:
                yhat = 1
            else:
                yhat = 0
            
            # Display the label above the image
            ax.set_title(f"{self._Y[random_index,0]},{yhat}")
            ax.set_axis_off()
        fig.suptitle("Label, yhat", fontsize=16)
        plt.show()

if __name__ == "__main__":
    handwritten = HandWrittenDigitsNN()
    handwritten.BuildModel()
    handwritten.Predict()