import argparse, numpy, h5py, tensorflow as tf, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, Conv2DTranspose, UpSampling2D, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
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
    _X_cv: numpy.array = None
    _Y_cv: numpy.array = None
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
    _grayscale: bool = None
    def __init__(self, name:str, grayscale:bool, path, input_shape, batch_size:int, learning_rate:float):
        self._name = name
        self._grayscale = grayscale
        self._input_shape = input_shape
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._model_path = path
        self._PrepareData()
        self._circuit_breaker = CreateCircuitBreakerCallback("val_loss", "min", 9)
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

        L1 Conv output dimension:
        Input: 64x64x3
        Filter: 8 4x4x3 s:1
        p = (f-1)/2 = 3/2 = 1
        Nc[l] = 8
        Nh[l] = floor((Nh[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(64 + 2 - 4)/1 + 1 = 63
        Nw[l] = floor((Nw[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(64 + 2 - 4)/1 + 1 = 63
        Output volume = Nh[l] x Nw[l] x Nc[l] = 63 x 63 x Nc[l]

        L3 MaxPoool2D output dimension:
        Input: 64x64x8
        Max Pooling: filter size:8, stride: 8
        p = (f-1)/2 = 7/2 = 3
        Nh[l] = floor((n+2p-f) / s) + 1 = floor((64+6-8) / 8) + 1 = floor(62/8) + 1 =  7+1 = 8
        Nw[l] = floor((n+2p-f) / s) + 1 = floor((64+6-8) / 8) + 1 = floor(62/8) + 1 =  7+1 = 8
        Output volume = Nh[l] x Nw[l] x Nc[l] = 8 x 8 x 8

        L4 Conv output dimension:
        Input: 8x8x8
        Filter: 16 2x2x8 s:1
        p = (f-1)/2 = 1/2 = 0
        Nc[l] = 16
        Nh[l] = floor((Nh[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(8 + 0 - 2)/1 + 1 = 7
        Nw[l] = floor((Nw[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(8 + 0 - 2)/1 + 1 = 7
        Output volume = Nh[l] x Nw[l] x Nc[l] = 7 x 7 x Nc[l]
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
                    Dense(self._classes, kernel_regularizer=l2(0.01)) # Linear activation ("pass-through") if not specified. Decrease to fix high bias; Increase to fix high variance.
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
            tensorboard = CreateTensorBoardCallback(self._name) # Create a new folder with current timestamp
            callbacks=[tensorboard]
            if use_circuit_breaker:
                callbacks.append(self._circuit_breaker)
            history = self._model.fit(self._train_dataset, epochs=epochs, shuffle=True, validation_data=self._validation_dataset, validation_freq=1, callbacks=callbacks) # shuffle: Boolean, whether to shuffle the training data before each epoch. This argument is ignored when x is a generator or a tf.data.Dataset.
            self._trained = True
            PlotModelHistory(f"{self._name} Multi-class Classifier", history)
            if self._model_path:
                self._model.save(self._model_path)
                print(f"Model saved to {self._model_path}.")
        self._model.evaluate(self._X_test, self._Y_test)

    def PredictSign(self, path:str, truth:int, grayscale:float = True):
        img = image.load_img(path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = numpy.expand_dims(x, axis=0)
        x2 = x
        if grayscale and x2.shape[-1] == 3:
            x2 = tf.image.rgb_to_grayscale(x2)
        elif not grayscale and x2.shape[-1] == 1:
            x2 = self._convert_grayscale_to_rgb(x2)
        #print(f"Input image: {path}, shape: {x.shape}")
        prediction = self._model.predict(x2)
        print(f"Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = {prediction}")
        prediction = numpy.argmax(prediction)
        color = bcolors.OKGREEN if truth == prediction else bcolors.FAIL
        print(f"{color}Truth: {truth}, Class: {prediction}{bcolors.DEFAULT}")

    def _PrepareData(self):
        self._classes = 10

        # This dataset only has 6 classes: [0:5] and is RGB. The Y is a single-value vector - SparseCategoricalCrossEntropy.
        train_dataset = h5py.File('data/SignsLanguage/train_signs.h5', "r")
        test_dataset = h5py.File('data/SignsLanguage/test_signs.h5', "r")
        _X_train = numpy.array(train_dataset["train_set_x"][:]) # (1080, 64, 64, 3)
        _Y_train = self._convert_labels_to_one_hot(numpy.array(train_dataset["train_set_y"][:])) # (1080,)

        _X_test = numpy.array(test_dataset["test_set_x"][:]) # (120, 64, 64, 3)
        _Y_test = self._convert_labels_to_one_hot(numpy.array(test_dataset["test_set_y"][:])) # (120,)

        if self._grayscale:
            _X_train = numpy.asarray([self._convert_rgb_to_grayscale(tf.convert_to_tensor(i)) for i in _X_train])
            _X_test = numpy.asarray([self._convert_rgb_to_grayscale(tf.convert_to_tensor(i)) for i in _X_test])

        # https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset
        # This dataset only has 10 classes: [0:9] and is grayscale.  The Y is a one-hot vector - CategoricalCrossEntropy.
        X = numpy.load("data/SignsLanguage/X.npy") # (2062, 64, 64)
        Y = numpy.load("data/SignsLanguage/Y.npy") # (2062, 10)
        X = X[..., numpy.newaxis] # Add a single grayacale channel
        if not self._grayscale:
            X = numpy.asarray([self._convert_grayscale_to_rgb(tf.convert_to_tensor(i)) for i in X])

        # Add the 2 sources of dataset before splitting them up to 3 datases - train/validation/test
        X_dataset = numpy.concatenate((numpy.concatenate((_X_train, X), axis=0), _X_test), axis=0)
        Y_dataset = numpy.concatenate((numpy.concatenate((_Y_train, Y), axis=0), _Y_test), axis=0)
        # X1: (1080, 64, 64, 3), X2: (120, 64, 64, 3), X: (2062, 64, 64, 3), total: (3262, 64, 64, 3)
        print(f"X1: {_X_train.shape}, X2: {_X_test.shape}, X: {X.shape}, total: {X_dataset.shape}")
        # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
        self._X_train, x_, self._Y_train, y_ = train_test_split(X_dataset, Y_dataset, test_size=0.30, random_state=1)

        # Split the 40% subset above into two: one half for cross validation and the other for the test set
        self._X_cv, self._X_test, self._Y_cv, self._Y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

        # Delete temporary variables
        del x_, y_

        print(f"X_train: {self._X_train.shape}, Y_train: {self._Y_train.shape}, X_cv: {self._X_cv.shape}, Y_cv: {self._Y_cv.shape}, X_test: {self._X_test.shape}, Y_test: {self._Y_test.shape}")
        # X_train: (2283, 64, 64, 3), Y_train: (2283, 10), X_cv: (489, 64, 64, 3), Y_cv: (489, 10), X_test: (490, 64, 64, 3), Y_test: (490, 10)

        self._train_dataset = tf.data.Dataset.from_tensor_slices((self._X_train, self._Y_train)).shuffle(self._Y_train.shape[0], reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self._validation_dataset = tf.data.Dataset.from_tensor_slices((self._X_cv, self._Y_cv)).shuffle(self._Y_cv.shape[0], reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        print (f"number of training examples = {self._X_train.shape[0]}")
        print (f"number of test examples = {self._X_test.shape[0]}")
        print (f"X_train shape: {self._X_train.shape}")
        print (f"Y_train shape: {self._Y_train.shape}")
        print (f"X_cv shape: {self._X_cv.shape}")
        print (f"Y_cv shape: {self._Y_cv.shape}")
        print (f"X_test shape: {self._X_test.shape}")
        print (f"Y_test shape: {self._Y_test.shape}")
        print(f"Class#: {self._classes}")
        print(f"Y_train: {self._Y_train[:10]}")
        print(f"Y_cv: {self._Y_cv[:10]}")
        print(f"Y_test: {self._Y_test[:10]}")
        """
        images_iter = iter(self._X_train)
        labels_iter = iter(self._Y_train)
        plt.figure(figsize=(10, 10))
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(next(images_iter).astype("uint8"))
            plt.title(next(labels_iter).astype("uint8"))
            plt.axis("off")
        plt.show()
        """
    def _convert_grayscale_to_rgb(self, image):
        # image is a 2D or 3D tensor (height, width, 1)
        # Replicate the single channel across 3 channels
        return tf.image.grayscale_to_rgb(image)

    def _convert_rgb_to_grayscale(self, image):
        # image is a 2D or 3D tensor (height, width, 3)
        return tf.image.rgb_to_grayscale(image)

    def _convert_labels_to_one_hot(self, data):
        # numpy.eye(): Return a 2-D array with ones on the diagonal and zeros elsewhere.
        return numpy.eye(self._classes)[data.reshape(-1)]

def ExamineRGBDataset():
    print(f"\n=== {ExamineRGBDataset.__name__} ===")
    train_dataset = h5py.File('data/SignsLanguage/train_signs.h5', "r")
    test_dataset = h5py.File('data/SignsLanguage/test_signs.h5', "r")
    X = numpy.load("data/SignsLanguage/X.npy") # (2062, 64, 64)
    Y = numpy.load("data/SignsLanguage/Y.npy") # (2062, 10)
    X = X[..., numpy.newaxis]
    #X_rgb = [] #numpy.asarray([tf.image.grayscale_to_rgb(i) for i in X]) # numpy.apply_along_axis(tf.image.grayscale_to_rgb, axis=0, arr = X)
    X_rgb = numpy.asarray([tf.image.grayscale_to_rgb(tf.convert_to_tensor(i)) for i in X])

    _X_train1 = numpy.array(train_dataset["train_set_x"][:]) # (1080, 64, 64, 3)
    _Y_train1 = numpy.array(train_dataset["train_set_y"][:]) # (1080,)
    _X_test1 = numpy.array(test_dataset["test_set_x"][:]) # (120, 64, 64, 3)
    _Y_test1 = numpy.array(test_dataset["test_set_y"][:]) # (120,)
    # _X_train1: (1080, 64, 64, 3), Y_train1: (1080,), X_test1: (120, 64, 64, 3), Y_test1: (120,), X: (2062, 64, 64), Y: (2062, 10)
    print(f"_X_train1: {_X_train1.shape}, Y_train1: {_Y_train1.shape}, X_test1: {_X_test1.shape}, Y_test1: {_Y_test1.shape}, X: {X.shape}, Y: {Y.shape}, X_rgb: {X_rgb.shape}")
    _Y_train2 = numpy.eye(10)[_Y_train1.reshape(-1)]
    _Y_test2 = numpy.eye(10)[_Y_test1.reshape(-1)]
    print(f"Y_train2: {_Y_train2.shape}, Y_test2: {_Y_test2.shape}") # Y_train2: (1080, 10), Y_test2: (120, 10)
    print("Y_train1[:10]:")
    print(_Y_train1[:10])
    print("Y_train2[:10]:")
    print(_Y_train2[:10])

def ExamineGrayscaleDataset():
    print(f"\n=== {ExamineGrayscaleDataset.__name__} ===")
    train_dataset = h5py.File('data/SignsLanguage/train_signs.h5', "r")
    test_dataset = h5py.File('data/SignsLanguage/test_signs.h5', "r")
    X = numpy.load("data/SignsLanguage/X.npy") # (2062, 64, 64)
    Y = numpy.load("data/SignsLanguage/Y.npy") # (2062, 10)
    X = X[..., numpy.newaxis]

    _X_train1 = numpy.array(train_dataset["train_set_x"][:]) # (1080, 64, 64, 3)
    _Y_train1 = numpy.array(train_dataset["train_set_y"][:]) # (1080,)
    _X_test1 = numpy.array(test_dataset["test_set_x"][:]) # (120, 64, 64, 3)
    _Y_test1 = numpy.array(test_dataset["test_set_y"][:]) # (120,)
    _X_train1 = numpy.asarray([tf.image.rgb_to_grayscale(tf.convert_to_tensor(i)) for i in _X_train1])
    _X_test1 = numpy.asarray([tf.image.rgb_to_grayscale(tf.convert_to_tensor(i)) for i in _X_test1])
    # _X_train1: (1080, 64, 64, 3), Y_train1: (1080,), X_test1: (120, 64, 64, 3), Y_test1: (120,), X: (2062, 64, 64), Y: (2062, 10)
    print(f"_X_train1: {_X_train1.shape}, Y_train1: {_Y_train1.shape}, X_test1: {_X_test1.shape}, Y_test1: {_Y_test1.shape}, X: {X.shape}, Y: {Y.shape}")
    _Y_train2 = numpy.eye(10)[_Y_train1.reshape(-1)]
    _Y_test2 = numpy.eye(10)[_Y_test1.reshape(-1)]
    print(f"Y_train2: {_Y_train2.shape}, Y_test2: {_Y_test2.shape}") # Y_train2: (1080, 10), Y_test2: (120, 10)
    print("Y_train1[:10]:")
    print(_Y_train1[:10])
    print("Y_train2[:10]:")
    print(_Y_train2[:10])

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Signs Language multi-class classifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    parser.add_argument('-g', '--grayscale', action='store_true', help='Use grayscale model')
    args = parser.parse_args()

    ExamineRGBDataset()
    ExamineGrayscaleDataset()
    model = f"models/SignsLanguageDigits_{'grayscale' if args.grayscale else 'RGB'}.keras"
    print(f"model: {model}")
    signs = SignsLanguageDigits("SignsLanguageDigits", args.grayscale, model , (64, 64, 1 if args.grayscale else 3), 32, 0.00015)
    signs.BuildModel()
    #InitializeGPU()
    signs.TrainModel(500, False, args.retrain)
    signs.PredictSign("images/my_handsign0.jpg", 2, args.grayscale)
    signs.PredictSign("images/my_handsign1.jpg", 1, args.grayscale)
    signs.PredictSign("images/my_handsign2.jpg", 3, args.grayscale)
    signs.PredictSign("images/my_handsign3.jpg", 5, args.grayscale)
    signs.PredictSign("images/my_handsign4.jpg", 5, args.grayscale)