import argparse, numpy, logging, warnings, pickle
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotModelHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.autograph.set_verbosity(0)

class HandWrittenDigitsNN():
    _X = None
    _Y = None
    _model: Sequential = None
    _model_path: str = None
    _learning_rate: float = None
    def __init__(self, path, learning_rate:float):
        self._learning_rate = learning_rate
        self._model_path = path
        InitializeGPU()
        self._prepare_data()
        self._visualize_data()
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._model = tf.keras.models.load_model(self._model_path)

    def _prepare_data(self):
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
        self._Y = numpy.load("data/y.npy", allow_pickle=True)
        self._X = self._X[0:1000]
        self._Y = self._Y[0:1000]

    def _visualize_data(self):
        print(f"\n=== {self._visualize_data.__name__} ===")
        m, n = self._X.shape
        fig, axes = plt.subplots(8,8, constrained_layout=True, figsize=(20, 20)) # figsize = (width, height)
        # Use tight_layout with h_pad to adjust vertical padding
        # Adjust h_pad for more/less vertical space
        fig.tight_layout(h_pad=2.0, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]
        for i, ax in enumerate(axes.flat):
            # Select random indices
            random_index = rng.integers(m, size=1)
            
            # Select rows corresponding to the random indices and
            # reshape the image
            X_random_reshaped = self._X[random_index].reshape((20,20)).T
            
            # Display the image
            ax.imshow(X_random_reshaped, cmap='gray')
            
            # Display the label above the image
            ax.set_title(self._Y[random_index,0][0], fontsize=20, pad=0) # The offset of the title from the top of the Axes, in points.
            ax.set_axis_off()
        fig.suptitle("Label", fontsize=22, fontweight="bold")
        plt.show()

    def BuildTrainModel(self, epochs:int, retrain: bool = False):
        print(f"\n=== {self.BuildTrainModel.__name__} ===")
        new_model:bool = not self._model
        if new_model:
            self._model = Sequential(
                [               
                    Input(shape=(400,)),    #specify input size
                    Dense(25, activation='sigmoid', name="L1"), # Densely connected, or fully connected
                    Dense(15, activation='sigmoid', name="L2"),
                    Dense(1, name="L3"),
                ], name = "HandWrittenDigits" 
            )
            self._model.compile(
                loss=BinaryCrossentropy(from_logits=True),  # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to sigmoid activation which is typically used for binary classification
                optimizer=Adam(self._learning_rate), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
            )
            self._model.summary()
            plot_model(
                self._model,
                to_file="output/HandWrittenDigitsNN.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB", # rankdir argument passed to PyDot, a string specifying the format of the plot: "TB" creates a vertical plot; "LR" creates a horizontal plot.
                expand_nested=True,
                show_layer_activations=True)
        if new_model or retrain:
            history = self._model.fit(self._X, self._Y, epochs=epochs, shuffle=True)
            PlotModelHistory("Hand Written Digits NN", history)
            if self._model_path:
                self._model.save(self._model_path)
                print(f"Model saved to {self._model_path}.")

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

        fig, axes = plt.subplots(8,8, constrained_layout=True, figsize=(20, 20)) # figsize = (width, height)
        # Use tight_layout with h_pad to adjust vertical padding
        # Adjust h_pad for more/less vertical space
        fig.tight_layout(h_pad=2.0, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]
        print(f"_Y: {self._Y.shape}")
        for i, ax in enumerate(axes.flat):
            # Select random indices
            random_index = rng.integers(m, size=1)
            
            # Select rows corresponding to the random indices and
            # reshape the image
            X_random_reshaped = self._X[random_index].reshape((20,20)).T
            
            # Display the image
            ax.imshow(X_random_reshaped, cmap='gray')
            
            # Predict using the Neural Network
            prediction = self._model.predict(self._X[random_index].reshape(1,400))
            yhat = int(prediction.item() >= 0.5)
            # Display the label above the image
            ax.set_title(f"{self._Y[random_index,0][0]},{yhat}", fontsize=20, pad=0)
            ax.set_axis_off()
        fig.suptitle("Label vs Prediction", fontsize=22, fontweight="bold")
        plt.show()

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Hand written digits binary classifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()
    handwritten = HandWrittenDigitsNN("models/HandwrittenDigits.keras", 0.001)
    handwritten.BuildTrainModel(100, args.retrain)
    handwritten.Predict()