import argparse, numpy, h5py, tensorflow as tf, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.pyplot import imshow
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, Conv2DTranspose, UpSampling2D, Normalization
from tensorflow.keras.optimizers import Adam
from utils.TrainingMetricsPlot import PlotModelHistory
from utils.TrainingUtils import CreateTensorBoardCallback, CreateCircuitBreakerCallback
from utils.GPU import InitializeGPU
from utils.TermColour import bcolors
class SignsLanguageDigits():
    """
    A convolution NN which differentiates among 6 sign language digits.
    """
    _input_shape = None
    _classes: int = None
    _X_train: numpy.array = None
    _Y_train: numpy.array = None
    _X_test: numpy.array = None
    _Y_test: numpy.array = None
    _model = None
    _model_path:str = None
    _circuit_breaker = None
    _batch_size: int = None
    _learning_rate: float = None
    _train_dataset = None
    _validation_dataset = None
    _trained:bool = None
    _name:str = None
    def __init__(self, name:str, path, input_shape, batch_size:int, learning_rate:float):
        self._name = name
        self._input_shape = input_shape
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._model_path = path
        self._PrepareData()
        self._circuit_breaker = CreateCircuitBreakerCallback("val_loss", "min", 7)
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._model = load_model(self._model_path)
            self._trained = True

    def BuildModel(self):
        """
        Implements the forward propagation for the model:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
        
        Note that for simplicity and grading purposes, you'll hard-code some values such as the stride and kernel (filter) sizes. 
        Normally, functions should take these values as function parameters.
        
        Arguments:
        input_img -- input dataset, of shape (input_shape)

        Returns:
        model -- TF Keras model (object containing the information for the entire training process) 
        """
        if not self._model:
            self._model = Sequential([
                    Input(shape=self._input_shape),
                    Normalization(axis=-1),
                    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
                    Conv2D(8, (4,4), strides=(1,1), padding="same", name="L1"),
                    BatchNormalization(axis=-1), # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
                    ## ReLU
                    ReLU(name="L2"),
                    Dropout(0.3),
                    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
                    MaxPool2D((8,8), strides=(8,8), padding="same", name="L3"),
                    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
                    Conv2D(16, (2,2), strides=(1,1), padding="same", name="L4"),
                    BatchNormalization(axis=-1), # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
                    ## ReLU
                    ReLU(name="L5"),
                    Dropout(0.3),
                    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
                    MaxPool2D((4,4), strides=(4,4), padding="same", name="L6"),
                    ## Flatten layer
                    Flatten(),
                    ## Dense layer with 1 unit for output & 'sigmoid' activation
                    Dense(6)   # Linear activation ("pass-through") if not specified. Since the labels are 6 categories, use CategoricalCrossentropy
                ])
            self._model.compile(
                    loss=CategoricalCrossentropy(from_logits=True), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
                    optimizer=Adam(learning_rate=self._learning_rate), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                    metrics=['accuracy']
                )
            self._model.summary()
            plot_model(
                self._model,
                to_file=f"output/{self._name}.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
            
    def TrainModel(self, epochs:int, use_circuit_breaker:bool = False, retrain: bool = False):
        if not self._trained or retrain:
            tensorboard = CreateTensorBoardCallback("SignsLanguageDigits") # Create a new folder with current timestamp
            callbacks=[tensorboard]
            if use_circuit_breaker:
                callbacks.append(self._circuit_breaker)
            history = self._model.fit(self._train_dataset, epochs=epochs, shuffle=True, validation_data=self._validation_dataset, validation_freq=1, callbacks=callbacks)
            self._trained = True
            PlotModelHistory("Signs Language Multi-class Classifier", history)
            if self._model_path:
                self._model.save(self._model_path)
                print(f"Model saved to {self._model_path}.")
        
    def PredictSign(self, path:str, truth:int):
        img = image.load_img(path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = numpy.expand_dims(x, axis=0)
        x2 = x 
        #print(f"Input image: {path}, shape: {x.shape}")
        prediction = self._model.predict(x2)
        print(f"Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = {prediction}")
        prediction = numpy.argmax(prediction)
        color = bcolors.OKGREEN if truth == prediction else bcolors.FAIL
        print(f"{color}Truth: {truth}, Class: {prediction}{bcolors.DEFAULT}")

    def _PrepareData(self):
        train_dataset = h5py.File('data/train_signs.h5', "r")
        self._X_train = numpy.array(train_dataset["train_set_x"][:]) # your train set features
        self._Y_train = numpy.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('data/test_signs.h5', "r")
        self._X_test = numpy.array(test_dataset["test_set_x"][:]) # your test set features
        self._Y_test = numpy.array(test_dataset["test_set_y"][:]) # your test set labels

        self._classes = numpy.array(test_dataset["list_classes"][:]) # the list of classes

        self._Y_train = self._Y_train.reshape((1, self._Y_train.shape[0]))
        self._Y_test = self._Y_test.reshape((1, self._Y_test.shape[0]))

        # Reshape
        self._convert_labels_to_one_hot()
        self._train_dataset = tf.data.Dataset.from_tensor_slices((self._X_train, self._Y_train)).shuffle(self._Y_train.shape[0], reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self._validation_dataset = tf.data.Dataset.from_tensor_slices((self._X_test, self._Y_test)).shuffle(self._Y_train.shape[0], reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        print (f"number of training examples = {self._X_train.shape[0]}")
        print (f"number of test examples = {self._X_test.shape[0]}")
        print (f"X_train shape: {self._X_train.shape}")
        print (f"Y_train shape: {self._Y_train.shape}")
        print (f"X_test shape: {self._X_test.shape}")
        print (f"Y_test shape: {self._Y_test.shape}")
        print(f"Class#: {self._classes} {self._classes.shape}")
        print(f"Y_train: {self._Y_train[:10]}")
        print(f"Y_test: {self._Y_test[:10]}")
        """
        images_iter = iter(self._X_train)
        labels_iter = iter(self._Y_train)
        plt.figure(figsize=(10, 10))
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(next(images_iter).numpy().astype("uint8"))
            plt.title(next(labels_iter).numpy().astype("uint8"))
            plt.axis("off")
        """

    def _convert_labels_to_one_hot(self):
        self._Y_train = numpy.eye(len(self._classes))[self._Y_train.reshape(-1)].T
        self._Y_test = numpy.eye(len(self._classes))[self._Y_test.reshape(-1)].T
        self._Y_train = self._Y_train.T
        self._Y_test = self._Y_test.T
        
if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Signs Language multi-class classifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    signs = SignsLanguageDigits("SignsLanguageDigits", "models/SignsLanguageDigits.keras", (64, 64, 3), 32, 0.00015)
    signs.BuildModel()
    #InitializeGPU()
    signs.TrainModel(500, False, args.retrain)
    signs.PredictSign("images/my_handsign0.jpg", 2)
    signs.PredictSign("images/my_handsign1.jpg", 1)
    signs.PredictSign("images/my_handsign2.jpg", 3)
    signs.PredictSign("images/my_handsign3.jpg", 5)
    signs.PredictSign("images/my_handsign4.jpg", 5)