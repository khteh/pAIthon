import argparse, numpy, h5py, tensorflow as tf, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from utils.TrainingMetricsPlot import PlotModelHistory
from utils.GPU import InitializeGPU
class SignsLanguageDigits():
    """
    A convolution NN which differentiates among 6 sign language digits.
    """
    _X_train: numpy.array = None
    _Y_train: numpy.array = None
    _X_test: numpy.array = None
    _Y_test: numpy.array = None
    _model: tf.keras.Sequential = None
    _model_path:str = None
    _trained: bool = False
    def __init__(self, path):
        InitializeGPU()
        self._model_path = path
        self._prepare_data()
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._model = tf.keras.models.load_model(self._model_path)
            self._trained = True

    def _prepare_data(self):
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
        
    def BuildModel(self, rebuild: bool = False, learning_rate:float = 0.01):
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
        if self._model and not rebuild:
            return
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
                optimizer=Adam(learning_rate=learning_rate), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                metrics=['accuracy']
            )
        self._model.summary()

    def TrainEvaluate(self, rebuild: bool, epochs:int, batch_size:int):
        if self._model:
            if not self._trained or rebuild:
                train_dataset = tf.data.Dataset.from_tensor_slices((self._X_train, self._Y_train)).batch(batch_size)
                validation_dataset = tf.data.Dataset.from_tensor_slices((self._X_test, self._Y_test)).batch(batch_size)
                history = self._model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)
                PlotModelHistory("Signs Language multi-class classifier", history)
                self._trained = True
                if self._model_path:
                    self._model.save(self._model_path)
                    print(f"Model saved to {self._model_path}.")
        else:
            raise RuntimeError("Please build the model first by calling BuildModel()!")
        
    def PredictSign(self, path:str, truth:int):
        img = image.load_img(path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = numpy.expand_dims(x, axis=0)
        x = x/255.0
        x2 = x 
        print(f"Input image: {path}, shape: {x.shape}")
        imshow(img)
        prediction = self._model.predict(x2)
        print(f"Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = {prediction}")
        prediction = numpy.argmax(prediction)
        print(f"Truth: {truth}, Class: {prediction}")
        #assert truth == prediction

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Signs Language multi-class classifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    signs = SignsLanguageDigits("models/SignsLanguageDigits.keras")
    signs.BuildModel(args.retrain, 0.01)
    signs.TrainEvaluate(args.retrain, 100, 64)
    signs.PredictSign("images/my_handsign0.jpg", 2)
    signs.PredictSign("images/my_handsign1.jpg", 1)
    signs.PredictSign("images/my_handsign2.jpg", 3)
    signs.PredictSign("images/my_handsign3.jpg", 5)
    signs.PredictSign("images/my_handsign4.jpg", 5)