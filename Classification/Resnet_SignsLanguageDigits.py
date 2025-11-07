import argparse
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
        super().TrainModel(epochs, True, retrain)

    def _PreprocessData(self, data):
        """
        Each Keras Application expects a specific kind of input preprocessing. 
        For ResNet, call keras.applications.resnet_v2.preprocess_input on your inputs before passing them to the model. 
        resnet_v2.preprocess_input will scale input pixels between -1 and 1.
        """
        print(f"\n=== {self._PreprocessData.__name__} ===")
        return preprocess_input(data)

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
    signs = ResnetSignsLanguageDigits("ResnetSignsLanguageDigits", args.grayscale, model, (64, 64, 1 if args.grayscale else 3), 32, 0.0001)
    signs.BuildModel()
    #InitializeGPU()
    signs.TrainModel(400, args.retrain)
    signs.PredictSign("images/my_handsign0.jpg", 2, args.grayscale)
    signs.PredictSign("images/my_handsign1.jpg", 1, args.grayscale)
    signs.PredictSign("images/my_handsign2.jpg", 3, args.grayscale)
    signs.PredictSign("images/my_handsign3.jpg", 5, args.grayscale)
    signs.PredictSign("images/my_handsign4.jpg", 5, args.grayscale)