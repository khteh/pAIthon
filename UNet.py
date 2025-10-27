import os, numpy, pandas as pd, imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
"""
data too large to keep in git:
19M	CameraMask
507M	CameraRGB
"""
class UNet():
    _input_size = None
    _n_filters: int = None
    _n_classes: int = None
    _model: Model = None
    def __init__(self, input_size, n_filters:int, n_classes: int):
        self._input_size = input_size
        self._n_filters = n_filters
        self._n_classes = n_classes
    """
    The encoder is a stack of various Encoders:

    Each Encoder() is composed of 2 Conv2D layers with ReLU activations. We will apply Dropout, and MaxPooling2D to some Encoders, as you will verify in the following sections, specifically to the last two blocks of the downsampling.

    The function will return two tensors:

    next_layer: That will go into the next block.
    skip_connection: That will go into the corresponding decoding block.
    Note: If max_pooling=True, the next_layer will be the output of the MaxPooling2D layer, but the skip_connection will be the output of the previously applied layer(Conv2D or Dropout, depending on the case). Else, both results will be identical.
    """
    def Encoder(self, inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
        """
        Convolutional downsampling block
        
        Arguments:
            inputs -- Input tensor
            n_filters -- Number of filters for the convolutional layers
            dropout_prob -- Dropout probability
            max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
        Returns: 
            next_layer, skip_connection --  Next layer and skip connection outputs
        """
        conv = Conv2D(n_filters, # Number of filters
                    3,   # Kernel size   
                    activation="relu",
                    padding="same",
                    kernel_initializer='he_normal')(inputs)
        conv = Conv2D(n_filters, # Number of filters
                    3,   # Kernel size
                    activation="relu",
                    padding="same",
                    # set 'kernel_initializer' same as above
                    kernel_initializer='he_normal')(conv)
        
        # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
        if dropout_prob > 0:
            conv = Dropout(dropout_prob)(conv)
            
        # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
        if max_pooling:
            next_layer = MaxPooling2D(2)(conv)
        else:
            next_layer = conv
            
        skip_connection = conv
        return next_layer, skip_connection

    def Decoder(self, expansive_input, contractive_input, n_filters=32):
        """
        Convolutional upsampling block
        
        Arguments:
            expansive_input -- Input tensor from previous layer
            contractive_input -- Input tensor from previous skip layer
            n_filters -- Number of filters for the convolutional layers
        Returns: 
            conv -- Tensor output
        """
        up = Conv2DTranspose(
                    n_filters,    # number of filters
                    3,    # Kernel size
                    strides=(2,2),
                    padding="same")(expansive_input)
        
        # Merge the previous output and the contractive_input
        merge = concatenate([up, contractive_input], axis=3)
        conv = Conv2D(n_filters,   # Number of filters
                    3,     # Kernel size
                    activation="relu",
                    padding="same",
                    kernel_initializer='he_normal')(merge)
        conv = Conv2D(n_filters,  # Number of filters
                    3,   # Kernel size
                    activation="relu",
                    padding="same",
                    # set 'kernel_initializer' same as above
                    kernel_initializer='he_normal')(conv)
        return conv

    def BuildModel(self):
        """
        Unet model
        
        Arguments:
            input_size -- Input shape 
            n_filters -- Number of filters for the convolutional layers
            n_classes -- Number of output classes
        Returns: 
            model -- tf.keras.Model
        """
        inputs = Input(self._input_size)
        # Contracting Path (encoding)
        # Add a Encoder with the inputs of the unet_ model and n_filters
        cblock1 = self.Encoder(inputs, self._n_filters)
        # Chain the first element, [0], of the output of each block to be the input of the next Encoder. 
        # Double the number of filters at each new step
        cblock2 = self.Encoder(cblock1[0], self._n_filters * 2)
        cblock3 = self.Encoder(cblock2[0], self._n_filters * 4)
        cblock4 = self.Encoder(cblock3[0], self._n_filters * 8, dropout_prob=0.3) # Include a dropout_prob of 0.3 for this layer
        # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
        cblock5 = self.Encoder(cblock4[0], self._n_filters * 16, dropout_prob=0.3, max_pooling=None)
        
        # Expanding Path (decoding)
        # Add the first Decoder.
        # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
        ublock6 = self.Decoder(cblock5[0], cblock4[1],  self._n_filters * 8)
        # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
        # Note that you must use the second element, [1], of the contractive block i.e before the maxpooling layer. 
        # At each step, use half the number of filters of the previous block 
        ublock7 = self.Decoder(ublock6, cblock3[1],  self._n_filters * 4)
        ublock8 = self.Decoder(ublock7, cblock2[1],  self._n_filters * 2)
        ublock9 = self.Decoder(ublock8, cblock1[1],  self._n_filters)

        conv9 = Conv2D(self._n_filters,
                    3,
                    activation='relu',
                    padding='same',
                    # set 'kernel_initializer' same as above exercises
                    kernel_initializer='he_normal')(ublock9)

        # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
        conv10 = Conv2D(self._n_classes, 1, padding="same")(conv9)
        
        self._model = tf.keras.Model(inputs=inputs, outputs=conv10)
        # In semantic segmentation, you need as many masks as you have object classes. In the dataset you're using, each pixel in every mask has been assigned a single integer probability that it belongs to a certain class, from 0 to num_classes-1. The correct class is the layer with the higher probability.
        # This is different from categorical crossentropy, where the labels should be one-hot encoded (just 0s and 1s). Here, you'll use sparse categorical crossentropy as your loss function, to perform pixel-wise multiclass prediction. Sparse categorical crossentropy is more efficient than other loss functions when you're dealing with lots of classes.
        self._model.compile(optimizer=Adam(0.01),
                    loss=SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        self._model.summary()
        plot_model(
            self._model,
            to_file="output/UNet.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="LR", # rankdir argument passed to PyDot, a string specifying the format of the plot: "TB" creates a vertical plot; "LR" creates a horizontal plot.
            expand_nested=True,
            show_layer_activations=True)

if __name__ == "__main__":
    img_height = 96
    img_width = 128
    num_channels = 3
    unet = UNet((img_height, img_width, num_channels), 32, 23)
    unet.BuildModel()