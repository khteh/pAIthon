import argparse, h5py, numpy, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, GlobalAveragePooling2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Normalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotModelHistory
from .SignsLanguageDigits import SignsLanguageDigits
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

class ResnetSignsLanguageDigits(SignsLanguageDigits):
    """
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
    Very deep "plain" networks don't work in practice because vanishing gradients make them hard to train.
    Skip connections help address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function.
    There are two main types of blocks: The identity block and the convolutional block.
    Very deep Residual Networks are built by stacking these blocks together.    
    """
    def BuildModel(self):
        """
        The details of this ResNet-50 model are:
        - Zero-padding pads the input with a pad of (3,3)
        - Stage 1:
            - The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). 
            - BatchNorm is applied to the 'channels' axis of the input.
            - MaxPooling uses a (3,3) window and a (2,2) stride.
        - Stage 2:
            - The convolutional block uses three sets of filters of size [64,64,256], "f" is 3, and "s" is 1.
            - The 2 identity blocks use three sets of filters of size [64,64,256], and "f" is 3.
        - Stage 3:
            - The convolutional block uses three sets of filters of size [128,128,512], "f" is 3 and "s" is 2.
            - The 3 identity blocks use three sets of filters of size [128,128,512] and "f" is 3.
        - Stage 4:
            - The convolutional block uses three sets of filters of size [256, 256, 1024], "f" is 3 and "s" is 2.
            - The 5 identity blocks use three sets of filters of size [256, 256, 1024] and "f" is 3.
        - Stage 5:
            - The convolutional block uses three sets of filters of size [512, 512, 2048], "f" is 3 and "s" is 2.
            - The 2 identity blocks use three sets of filters of size [512, 512, 2048] and "f" is 3.
        - The 2D Average Pooling uses a window (pool_size) of shape (2,2).
        - The 'flatten' layer doesn't have any hyperparameters.
        - The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation.

        Stage-wise implementation of the architecture of the popular ResNet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        if not self._model:
            model = ResNet152V2(input_shape = self._input_shape, include_top=False)
            model.trainable = True
            # Let's take a look to see how many layers are in the base model
            print("Number of layers in the base model: ", len(model.layers))

            # Fine-tune from this layer onwards
            # Where the final layers actually begin is a bit arbitrary, so feel free to play around with this number a bit. The important takeaway is that the later layers are the part of your network that contain the fine details (pointy ears, hairy tails) that are more specific to your problem.
            NUM_LAYERS_TO_TUNE = 30
            fine_tune_at = len(model.layers) - NUM_LAYERS_TO_TUNE

            # Freeze all the layers before the `fine_tune_at` layer
            for layer in model.layers[:fine_tune_at]:
                layer.trainable = False

            print(f"Layers to fine-tune: {[layer.name for layer in model.layers[fine_tune_at:]]}")

            x = model.output

            # add the new Multi-class classification layers
            # use global avg pooling to summarize the info in each channel
            x = GlobalAveragePooling2D(name="AveragePooling")(x)
            x = BatchNormalization()(x)

            # include dropout with probability of 0.2 to avoid overfitting
            x = Dropout(0.2, name="FinalDropout")(x)
                
            # use a prediction layer with one neuron (as a binary classifier only needs one)
            x = Flatten()(x)
            outputs = Dense(self._classes, name="FinalOutput", activation="softmax", kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=l2(0.01))(x) # Decrease to fix high bias; Increase to fix high variance.

            # Create model
            self._model = Model(model.input, outputs)
            self._model.name = self._name
            self._model.compile(
                    loss=CategoricalCrossentropy(from_logits=False), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
                    optimizer=Adam(learning_rate=self._learning_rate * 0.1), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
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
            
    def _BuildBaseModel(self):
        """
        The details of this ResNet-50 model are:
        - Zero-padding pads the input with a pad of (3,3)
        - Stage 1:
            - The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). 
            - BatchNorm is applied to the 'channels' axis of the input.
            - MaxPooling uses a (3,3) window and a (2,2) stride.
        - Stage 2:
            - The convolutional block uses three sets of filters of size [64,64,256], "f" is 3, and "s" is 1.
            - The 2 identity blocks use three sets of filters of size [64,64,256], and "f" is 3.
        - Stage 3:
            - The convolutional block uses three sets of filters of size [128,128,512], "f" is 3 and "s" is 2.
            - The 3 identity blocks use three sets of filters of size [128,128,512] and "f" is 3.
        - Stage 4:
            - The convolutional block uses three sets of filters of size [256, 256, 1024], "f" is 3 and "s" is 2.
            - The 5 identity blocks use three sets of filters of size [256, 256, 1024] and "f" is 3.
        - Stage 5:
            - The convolutional block uses three sets of filters of size [512, 512, 2048], "f" is 3 and "s" is 2.
            - The 2 identity blocks use three sets of filters of size [512, 512, 2048] and "f" is 3.
        - The 2D Average Pooling uses a window (pool_size) of shape (2,2).
        - The 'flatten' layer doesn't have any hyperparameters.
        - The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation.

        Stage-wise implementation of the architecture of the popular ResNet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        if not self._model:
            base_model = ResNet152V2(input_shape = self._input_shape, include_top=False)
            # freeze the base model by making it non trainable
            base_model.trainable = False
            x = base_model.output

            # add the new Multi-class classification layers
            # use global avg pooling to summarize the info in each channel
            x = GlobalAveragePooling2D(name="AveragePooling")(x)
            x = BatchNormalization()(x)

            # include dropout with probability of 0.2 to avoid overfitting
            x = Dropout(0.3, name="FinalDropout")(x)
                
            # use a prediction layer with one neuron (as a binary classifier only needs one)
            x = Flatten()(x)
            outputs = Dense(self._classes, name="FinalOutput", activation="softmax", kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=l2(0.01))(x) # Decrease to fix high bias; Increase to fix high variance.

            # Create model
            self._model = Model(base_model.input, outputs)
            self._model.name = self._name
            self._model.trainable = True
            # Let's take a look to see how many layers are in the base model

            # Fine-tune from this layer onwards
            # Where the final layers actually begin is a bit arbitrary, so feel free to play around with this number a bit. The important takeaway is that the later layers are the part of your network that contain the fine details (pointy ears, hairy tails) that are more specific to your problem.
            NUM_LAYERS_TO_TUNE = 30
            fine_tune_at = len(model.layers) - NUM_LAYERS_TO_TUNE

            # Freeze all the layers before the `fine_tune_at` layer
            for layer in self._model.layers[:fine_tune_at]:
                layer.trainable = False

            print(f"Layers to fine-tune: {[layer.name for layer in self._model.layers[fine_tune_at:]]}")

            self._model.compile(
                    loss=CategoricalCrossentropy(from_logits=False), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
                    optimizer=Adam(learning_rate=self._learning_rate), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                    metrics=['accuracy']
                )
            #self._model.summary() Too long.
            plot_model(
                self._model,
                to_file=f"output/{self._name}.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)

    def TrainModel(self, epochs:int, retrain: bool = False):
        super().TrainModel(epochs, False, retrain)

    def _PreprocessData(self, data):
        """
        Each Keras Application expects a specific kind of input preprocessing. 
        For ResNet, call keras.applications.resnet_v2.preprocess_input on your inputs before passing them to the model. 
        resnet_v2.preprocess_input will scale input pixels between -1 and 1.
        
        data: A floating point numpy.array or a backend-native tensor, 3D or 4D with 3 color channels, with values in the range [0, 255]. 
              The preprocessed data are written over the input data if the data types are compatible. To avoid this behaviour, numpy.copy(x) can be used.

        Returns: Preprocessed array with type float32. The inputs pixel values are scaled between -1 and 1, sample-wise.
        """
        print(f"\n=== {self._PreprocessData.__name__} ===")
        assert 3 == data.shape[-1]
        return preprocess_input(data)

    def _undo_resnetv2_preprocess_np(self, data):
        out = (data + 1.0) * 127.5
        return numpy.clip(out, 0, 255).astype(numpy.uint8)

    def _undo_grayscale_to_rgb_np(self, data_rgb):
        gray = data_rgb[..., 0]   # works because R=G=B
        return numpy.expand_dims(gray, axis=-1)

    def _UnProcessData(self, data):
        rgb = self._undo_resnetv2_preprocess_np(data)
        gray = self._undo_grayscale_to_rgb_np(rgb)
        return gray, rgb
    
    def ShowGrayscaleDataset(self):
        print(f"\n=== {self.ShowGrayscaleDataset.__name__} ===")
        X = numpy.load("data/SignsLanguage/X.npy") # (2062, 64, 64)
        Y = numpy.load("data/SignsLanguage/Y.npy") # (2062, 10)
        X = X[..., numpy.newaxis] # Add a single grayacale channel
        print(f"X: {X.shape}, Y_test: {Y.shape}, X mean: {numpy.mean(X)}, [{numpy.min(X)}, {numpy.max(X)}]")
        fig, axes = plt.subplots(5, 5, constrained_layout=True, figsize=(20, 20)) # figsize = (width, height)
        # Use tight_layout with h_pad to adjust vertical padding
        # Adjust h_pad for more/less vertical space
        fig.tight_layout(h_pad=2.0, rect=[0, 0.03, 1, 0.95]) #[left, bottom, right, top]
        for i, ax in enumerate(axes.flat):
            # Select random indices
            index = rng.integers(X.shape[0], size=1)
            image = X[index][0, :, :, :] # Remove the first dimension (batch)
            #print(f"X: {numpy.mean(input)} [{numpy.min(input)}, {numpy.max(input)}]")
            print(f"image {image.shape}: mean: {numpy.mean(image)} [{numpy.min(image)}, {numpy.max(image)}]")
            # Display the image
            ax.imshow(image)
            ax.set_axis_off()
        fig.suptitle("Grayscale Dataset", fontsize=22, fontweight="bold")
        plt.show()
        return numpy.mean(X), numpy.min(X), numpy.max(X)
       
    def ShowGrayscaleConvertedToRGBDataset(self):
        print(f"\n=== {self.ShowGrayscaleConvertedToRGBDataset.__name__} ===")
        X = numpy.load("data/SignsLanguage/X.npy") # (2062, 64, 64)
        X = X[..., numpy.newaxis] # Add a single grayacale channel
        X = numpy.asarray([self._convert_grayscale_to_rgb(tf.convert_to_tensor(i)) for i in X])
        Y = numpy.load("data/SignsLanguage/Y.npy") # (2062, 10)
        print(f"X: {X.shape}, Y_test: {Y.shape}, X mean: {numpy.mean(X)}, [{numpy.min(X)}, {numpy.max(X)}]")
        fig, axes = plt.subplots(5, 5, constrained_layout=True, figsize=(20, 20)) # figsize = (width, height)
        # Use tight_layout with h_pad to adjust vertical padding
        # Adjust h_pad for more/less vertical space
        fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95]) #[left, bottom, right, top]
        for i, ax in enumerate(axes.flat):
            # Select random indices
            index = rng.integers(X.shape[0], size=1)
            image = X[index][0, :, :, :] # Remove the first dimension (batch)
            print(f"image {image.shape}: mean: {numpy.mean(image)} [{numpy.min(image)}, {numpy.max(image)}]")
            # Display the image
            ax.imshow(image)
            ax.set_axis_off()
        fig.suptitle("Grayscale Processed Dataset", fontsize=22, fontweight="bold")
        plt.show()
        return numpy.mean(X), numpy.min(X), numpy.max(X)

    def ShowGrayscaleConvertedToRGBProcessedDataset(self):
        print(f"\n=== {self.ShowGrayscaleConvertedToRGBProcessedDataset.__name__} ===")
        X = numpy.load("data/SignsLanguage/X.npy") # (2062, 64, 64)
        Y = numpy.load("data/SignsLanguage/Y.npy") # (2062, 10)
        X = X[..., numpy.newaxis] # Add a single grayacale channel
        X_rgb1 = numpy.asarray([self._convert_grayscale_to_rgb(tf.convert_to_tensor(i)) for i in X])
        print(f"X_rgb1: {X_rgb1.shape}, Y_test: {Y.shape}, X_rgb1 mean: {numpy.mean(X_rgb1)}, [{numpy.min(X_rgb1)}, {numpy.max(X_rgb1)}]")
        X_rgb = numpy.asarray([self._convert_grayscale_to_rgb(tf.convert_to_tensor(i)) for i in X]) * 255 # fit it again in the RGB scale , once it is done you can then directly apply the linear function
        print(f"X_rgb: {X_rgb.shape}, Y_test: {Y.shape}, X_rgb mean: {numpy.mean(X_rgb)}, [{numpy.min(X_rgb)}, {numpy.max(X_rgb)}]")
        X = self._PreprocessData(X_rgb)
        print(f"X: {X.shape}, Y_test: {Y.shape}, X mean: {numpy.mean(X)} [{numpy.min(X)},{numpy.max(X)}]")
        gray, rgb = self._UnProcessData(X)
        print(f"gray: {gray.shape}, gray mean: {numpy.mean(gray)}, [{numpy.min(gray)}, {numpy.max(gray)}]")
        print(f"rgb: {rgb.shape}, rgb mean: {numpy.mean(rgb)}, [{numpy.min(rgb)}, {numpy.max(rgb)}]")
        fig, axes = plt.subplots(5, 5, constrained_layout=True, figsize=(20, 20)) # figsize = (width, height)
        # Use tight_layout with h_pad to adjust vertical padding
        # Adjust h_pad for more/less vertical space
        fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95]) #[left, bottom, right, top]
        for i, ax in enumerate(axes.flat):
            # Select random indices
            index = rng.integers(rgb.shape[0], size=1)
            image = rgb[index][0, :, :, :] # Remove the first dimension (batch)
            print(f"image {image.shape}: mean: {numpy.mean(image)} [{numpy.min(image)}, {numpy.max(image)}]")
            # Display the image
            ax.imshow(image)
            ax.set_axis_off()
        fig.suptitle("Grayscale RGB Processed Dataset", fontsize=22, fontweight="bold")
        plt.show()
        return numpy.mean(rgb), numpy.min(rgb), numpy.max(rgb)

    def ShowRGBProcessedDataset(self):
        print(f"\n=== {self.ShowRGBProcessedDataset.__name__} ===")
        # This dataset only has 6 classes: [0:5] and is RGB. The Y is a single-value vector - SparseCategoricalCrossEntropy.
        train_dataset = h5py.File('data/SignsLanguage/train_signs.h5', "r")
        test_dataset = h5py.File('data/SignsLanguage/test_signs.h5', "r")
        X_train = numpy.array(train_dataset["train_set_x"][:]) # (1080, 64, 64, 3)
        Y_train = self._convert_labels_to_one_hot(numpy.array(train_dataset["train_set_y"][:])) # (1080,)

        X_test = numpy.array(test_dataset["test_set_x"][:]) # (120, 64, 64, 3)
        Y_test = self._convert_labels_to_one_hot(numpy.array(test_dataset["test_set_y"][:])) # (120,)

        print(f"_X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_train mean: {numpy.mean(X_train)}, [{numpy.min(X_train)}, {numpy.max(X_train)}]")
        X = self._PreprocessData(X_train)
        print(f"X: {X.shape}, X mean: {numpy.mean(X)} [{numpy.min(X)},{numpy.max(X)}]")
        gray, rgb = self._UnProcessData(X)
        print(f"gray: {gray.shape}, gray mean: {numpy.mean(gray)}, [{numpy.min(gray)}, {numpy.max(gray)}]")
        print(f"rgb: {rgb.shape}, rgb mean: {numpy.mean(rgb)}, [{numpy.min(rgb)}, {numpy.max(rgb)}]")

        fig, axes = plt.subplots(5, 5, constrained_layout=True, figsize=(20, 20)) # figsize = (width, height)
        # Use tight_layout with h_pad to adjust vertical padding
        # Adjust h_pad for more/less vertical space
        fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95]) #[left, bottom, right, top]
        for i, ax in enumerate(axes.flat):
            # Select random indices
            index = rng.integers(rgb.shape[0], size=1)
            image = rgb[index][0, :, :, :] # Remove the first dimension (batch)
            print(f"image {image.shape}: mean: {numpy.mean(image)} [{numpy.min(image)}, {numpy.max(image)}]")
            # Display the image
            ax.imshow(image)
            ax.set_axis_off()
        fig.suptitle("Grayscale RGB Processed Dataset", fontsize=22, fontweight="bold")
        plt.show()
        return numpy.mean(rgb), numpy.min(rgb), numpy.max(rgb)

def DataProcessingTests():
    print(f"\n=== {DataProcessingTests.__name__} ===")
    model = f"models/ResnetSignsLanguageDigits_{'grayscale' if args.grayscale else 'RGB'}.keras"
    signs = ResnetSignsLanguageDigits("ResnetSignsLanguageDigits", args.grayscale, model, (64, 64, 1 if args.grayscale else 3), 32, 0.0001)
    mean, min, max = signs.ShowGrayscaleDataset()
    assert 0.0 == min
    assert 1.0 == max
    mean, min, max = signs.ShowGrayscaleConvertedToRGBDataset()
    assert 0.0 == min
    assert 1.0 == max
    mean, min, max = signs.ShowGrayscaleConvertedToRGBProcessedDataset()
    assert min >= 0.0, f"min: {min}"
    assert max <= 255.0, f"max: {max}"
    assert mean >= min and mean <= max, f"mean: {mean}"
    mean, min, max = signs.ShowRGBProcessedDataset()
    assert min >= 0.0, f"min: {min}"
    assert max <= 255.0, f"max: {max}"
    assert mean >= min and mean <= max, f"mean: {mean}"

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Resnet Signs Language multi-class classifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    parser.add_argument('-g', '--grayscale', action='store_true', help='Use grayscale model')
    args = parser.parse_args()
    model = f"models/ResnetSignsLanguageDigits_{'grayscale' if args.grayscale else 'RGB'}.keras"
    signs = ResnetSignsLanguageDigits("ResnetSignsLanguageDigits", args.grayscale, model, (64, 64, 1 if args.grayscale else 3), 32, 0.00001)
    #DataProcessingTests()
    signs.BuildModel()
    #InitializeGPU()
    signs.TrainModel(1000, args.retrain)
    #signs.Evaluate()
    signs.PredictSign("images/my_handsign0.jpg", 2, args.grayscale)
    signs.PredictSign("images/my_handsign1.jpg", 1, args.grayscale)
    signs.PredictSign("images/my_handsign2.jpg", 3, args.grayscale)
    signs.PredictSign("images/my_handsign3.jpg", 5, args.grayscale)
    signs.PredictSign("images/my_handsign4.jpg", 5, args.grayscale)