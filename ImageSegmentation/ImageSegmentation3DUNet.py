import argparse, json, numpy, pandas as pd, imageio.v3 as iio, nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from keras import saving
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, concatenate, Input, Conv3D, MaxPooling3D, Dropout, Conv3DTranspose, BatchNormalization, Normalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from utils.GPU import InitializeGPU, SetMemoryLimit
from utils.TrainingMetricsPlot import PlotModelHistory
from utils.TrainingUtils import CreateTensorBoardCallback, CreateCircuitBreakerCallback
from .VolumeDataGenerator import VolumeDataGenerator
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
"""
data:
https://decathlon-10.grand-challenge.org/
"""
@saving.register_keras_serializable()
def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), 
                     epsilon=0.00001):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the tf.math.reduce_sum function.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """
    dice_numerator = 2 * tf.math.reduce_sum(tf.multiply(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)), axis=axis) + epsilon
    dice_denominator = tf.math.reduce_sum(y_true, axis=axis) + tf.math.reduce_sum(y_pred, axis=axis) + epsilon
    dice_coefficient = tf.math.reduce_mean(dice_numerator / dice_denominator)
    return dice_coefficient

@saving.register_keras_serializable()
def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the tf.math.reduce_sum function.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.     
    """
    dice_numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = tf.math.reduce_sum(y_true ** 2, axis=axis) + tf.math.reduce_sum(y_pred ** 2, axis=axis) + epsilon
    dice_coefficient = tf.math.reduce_mean(dice_numerator / dice_denominator)
    dice_loss = 1 - dice_coefficient
    return dice_loss

class ImageSegmentation3DUNet():
    """
    U-Net, a type of CNN designed for quick, precise image segmentation, and using it to predict a label for every single pixel in an image - in this case, an image from a self-driving car dataset. 
    This type of image classification is called semantic image segmentation. It's similar to object detection in that both ask the question: "What objects are in this image and where in the image are those objects located?," 
    but where object detection labels objects with bounding boxes that may include pixels that aren't part of the object, semantic image segmentation allows you to predict a precise mask for each object in the image by labeling each pixel in the image with its corresponding class. 
    The word “semantic” here refers to what's being shown, so for example the “Car” class is indicated below by the dark blue mask, and "Person" is indicated with a red mask.

    The "depth" of a U-Net is equal to the number of down-convolutions it uses including the very bottom of the U.
    """
    _path:str = None
    _model_path:str = None
    _input_shape = None
    _n_filters: int = None
    _n_classes: int = None
    _buffer_size: int = None
    _batch_size: int = None
    _train_dataset = None
    _val_dataset = None
    _learning_rate: float = None
    _model: Model = None
    _normalization: Normalization = None
    _circuit_breaker = None
    _train_generator = None
    _valid_generator = None
    def __init__(self, path:str, model_path:str, input_shape, n_filters:int, n_classes: int, buffer_size:int, batch_size:int, learning_rate:float):
        self._path = path
        self._model_path = model_path
        self._input_shape = input_shape
        self._n_filters = n_filters
        self._n_classes = n_classes
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._normalization = Normalization(axis=1)
        self._circuit_breaker = CreateCircuitBreakerCallback("val_dice_coefficient", "max", 7)
        self._PrepareData()
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._model = load_model(self._model_path)

    def Encoder(self, inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
        """
        Convolutional downsampling block
        The encoder is a stack of various Encoders:
        Each Encoder() is composed of 2 Conv3D layers with ReLU activations. We will apply Dropout, and MaxPooling2D to some Encoders, as you will verify in the following sections, specifically to the last two blocks of the downsampling.
        The #filters is specified for each depth and each layer within it as filters(i) = 32 * 2^i. i is the current depth. So for depth 0, filters(0) = 32.
        For layer-1 of depth-0, the #filters is filters(i) = 32 * 2^i * 2. i is the current depth. So for depth 0, layer 1: filters(i) = 32 * 2 = 64. Notice that the '*2' factor only applies to layer-1 but not layer-0.

        The function will return two tensors:

        next_layer: That will go into the next block.
        skip_connection: That will go into the corresponding decoding block.
        Note: If max_pooling=True, the next_layer will be the output of the MaxPooling2D layer, but the skip_connection will be the output of the previously applied layer(Conv3D or Dropout, depending on the case). Else, both results will be identical.
        
        Arguments:
            inputs -- Input tensor
            n_filters -- Number of filters for the convolutional layers
            dropout_prob -- Dropout probability
            max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
        Returns: 
            next_layer, skip_connection --  Next layer and skip connection outputs
        """
        conv = Conv3D(n_filters, # Number of filters
                    3,   # Kernel size   
                    activation="relu",
                    padding="same", data_format='channels_first',
                    kernel_initializer='he_normal')(inputs)
        conv = BatchNormalization(axis=1)(conv)
        conv = Conv3D(n_filters, # Number of filters
                    3,   # Kernel size
                    activation="relu",
                    padding="same", data_format='channels_first',
                    # set 'kernel_initializer' same as above
                    kernel_initializer='he_normal')(conv)
        conv = BatchNormalization(axis=1)(conv)
        
        # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
        if dropout_prob > 0:
            conv = Dropout(dropout_prob)(conv)
            
        # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
        next_layer = MaxPooling3D(2, data_format='channels_first')(conv) if max_pooling else conv
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
        up = Conv3DTranspose(
                    n_filters,    # number of filters
                    3,    # Kernel size
                    strides=(2,2,2), data_format='channels_first',
                    padding="same")(expansive_input)
        
        # Merge the previous output and the contractive_input
        merge = concatenate([up, contractive_input], axis=1)
        conv = Conv3D(n_filters,   # Number of filters
                    3,     # Kernel size
                    activation="relu",
                    padding="same", data_format='channels_first',
                    kernel_initializer='he_normal')(merge)
        conv = BatchNormalization(axis=1)(conv)
        conv = Conv3D(n_filters,  # Number of filters
                    3,   # Kernel size
                    activation="relu",
                    padding="same", data_format='channels_first',
                    # set 'kernel_initializer' same as above
                    kernel_initializer='he_normal')(conv)
        conv = BatchNormalization(axis=1)(conv)
        return conv

    def BuildTrainModel(self, epochs: int, steps_per_epoch:int, validation_steps:int, retrain:bool = False):
        """
        Unet model
        
        Arguments:
            input_shape -- Input shape 
            n_filters -- Number of filters for the convolutional layers
            n_classes -- Number of output classes
        Returns: 
            model -- tf.keras.Model
        """
        print(f"\n=== {self.BuildTrainModel.__name__} ===")
        new_model = not self._model
        if not self._model:
            inputs = Input(self._input_shape)
            #print(f"inputs: {tf.math.reduce_mean(inputs, axis=1)}")
            inputs = self._normalization(inputs)
            # Contracting Path (encoding)
            # Add a Encoder with the inputs of the unet_ model and n_filters
            cblock1 = self.Encoder(inputs, self._n_filters)
            # Chain the first element, [0], of the output of each block to be the input of the next Encoder. 
            # Double the number of filters at each new step
            cblock2 = self.Encoder(cblock1[0], self._n_filters * 2)
            cblock3 = self.Encoder(cblock2[0], self._n_filters * 4)
            cblock4 = self.Encoder(cblock3[0], self._n_filters * 8, dropout_prob=0.3) # Include a dropout_prob of 0.3 for this layer
            # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
            cblock5 = self.Encoder(cblock4[0], self._n_filters * 16, dropout_prob=0.3, max_pooling=False)
            
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

            conv9 = Conv3D(self._n_filters,
                        3,
                        activation='relu',
                        padding='same', data_format='channels_first',
                        # set 'kernel_initializer' same as above exercises
                        kernel_initializer='he_normal')(ublock9)
            conv9 = BatchNormalization(axis=1)(conv9)
            # Add a Conv3D layer with n_classes filter, kernel size of 1 and a 'same' padding
            conv10 = Conv3D(self._n_classes, 1, padding="same", data_format='channels_first')(conv9)
            conv10 = BatchNormalization(axis=1)(conv10)
            output = Activation("sigmoid")(conv10)
            self._model = tf.keras.Model(inputs=inputs, outputs=output)
            # In semantic segmentation, you need as many masks as you have object classes. In the dataset you're using, each pixel in every mask has been assigned a single integer probability that it belongs to a certain class, from 0 to num_classes-1. The correct class is the layer with the higher probability.
            # This is different from categorical crossentropy, where the labels should be one-hot encoded (just 0s and 1s). Here, you'll use sparse categorical crossentropy as your loss function, to perform pixel-wise multiclass prediction. Sparse categorical crossentropy is more efficient than other loss functions when you're dealing with lots of classes.
            self._model.compile(optimizer=Adam(self._learning_rate),
                        loss=soft_dice_loss,
                        metrics=[dice_coefficient])
            self._model.summary()
            plot_model(
                self._model,
                dpi=1000,
                to_file="output/ImageSegmentation3DUNetModel.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="LR", # rankdir argument passed to PyDot, a string specifying the format of the plot: "TB" creates a vertical plot; "LR" creates a horizontal plot.
                expand_nested=True,
                show_layer_activations=True)
        if new_model or retrain:
            tensorboard = CreateTensorBoardCallback("ImageSegmentation3DUNet") # Create a new folder with current timestamp
            # shuffle: Boolean, whether to shuffle the training data before each epoch. This argument is ignored when x is a generator or a tf.data.Dataset.
            history = self._model.fit(self._train_generator, shuffle=True,
                    epochs=epochs,
                    validation_data=self._valid_generator,
                    #steps_per_epoch=steps_per_epoch,
                    #validation_steps=validation_steps,
                    callbacks=[tensorboard])
            PlotModelHistory("ImageSegmentation 3D UNet", history)
            if self._model_path:
                self._model.save(self._model_path)
                print(f"Model saved to {self._model_path}.")

    def _display(self, images):
        titles = ['Input Image', 'True Mask', 'Predicted Mask']
        fig, axes = plt.subplots(len(images), len(titles), constrained_layout=True, figsize=(15, 20)) # figsize = (width, height)
        # rect=[0, 0, 1, 0.98] tells tight_layout to arrange the subplots within the bottom 98% of the figure's height, leaving the top 2% some space for the suptitle, for instance.
        fig.tight_layout(pad=0.1,rect=[0, 0, 1, 0.98]) #[left, bottom, right, top] Decrease the top boundary if the suptitle overlaps with the plots
        for i in range(len(images)):
            for j in range(len(titles)):
                axes[i][j].title.set_text(titles[j])
                axes[i][j].imshow(tf.keras.preprocessing.image.array_to_img(images[i][j]))
                axes[i][j].axis('off')
        plt.savefig(f"output/ImageSegmentation3DUNet.png")
        #plt.show()
        plt.close()

    def _create_mask(self, pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]            

    def _visualize_patch(self, X, y, title:str):
        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=[10, 5])
        fig.tight_layout(pad=1, rect=[0, 0, 1, 0.95]) #[left, bottom, right, top] Decrease the top boundary if the suptitle overlaps with the plots
        ax[0].imshow(X[:, :, 0], cmap='Greys_r')
        ax[0].set_yticks([])
        ax[0].set_xticks([])

        ax[1].imshow(y[:, :, 0], cmap='Greys_r')
        ax[1].set_yticks([])
        ax[1].set_xticks([])

        fig.suptitle(title, fontsize=22, fontweight="bold")
        plt.savefig(f"output/ImageSegmentation3DUNet_MRI_{title}.png")

    def show_predictions(self):
        """
        Displays the first image of each of the num batches.
        """
        print(f"\n=== {self.show_predictions.__name__} ===")
        image = numpy.array(nib.load(f"{self._path}/imagesTr/BRATS_001.nii.gz").get_fdata())
        label = numpy.array(nib.load(f"{self._path}/labelsTr/BRATS_001.nii.gz").get_fdata())
        X, y = self._get_sub_volume(image, label)
        X_norm = self._standardize(X)
        X_norm_with_batch_dimension = numpy.expand_dims(X_norm, axis=0)

        patch_pred = self._model.predict(X_norm_with_batch_dimension)
        # set threshold.
        threshold = 0.5

        # use threshold to get hard predictions
        patch_pred[patch_pred > threshold] = 1.0
        patch_pred[patch_pred <= threshold] = 0.0
        print("Patch and ground truth")
        self._visualize_patch(X_norm[0, :, :, :], y[2], "Patch and ground truth")
        plt.show()
        print("Patch and prediction")
        self._visualize_patch(X_norm[0, :, :, :], patch_pred[0, 2, :, :, :], "Patch and prediction")
        plt.show()

    def _PrepareData(self):
        """
        Our dataset is stored in the [NifTI-1 format](https://nifti.nimh.nih.gov/nifti-1/) and we will be using the [NiBabel library](https://github.com/nipy/nibabel) to interact with the files. Each training sample is composed of two separate files:

        The first file is an image file containing a 4D array of MR image in the shape of (240, 240, 155, 4). 
        -  The first 3 dimensions are the X, Y, and Z values for each point in the 3D volume, which is commonly called a voxel. 
        - The 4th dimension is the values for 4 different sequences
            - 0: FLAIR: "Fluid Attenuated Inversion Recovery" (FLAIR)
            - 1: T1w: "T1-weighted"
            - 2: t1gd: "T1-weighted with gadolinium contrast enhancement" (T1-Gd)
            - 3: T2w: "T2-weighted"

        The second file in each training example is a label file containing a 3D array with the shape of (240, 240, 155).  
        - The integer values in this array indicate the "label" for each voxel in the corresponding image files:
            - 0: background
            - 1: edema
            - 2: non-enhancing tumor
            - 3: enhancing tumor

        We have access to a total of 484 training images which we will be splitting into a training (80%) and validation (20%) dataset.

        Data Preprocessing using Patches:
        While our dataset is provided to us post-registration and in the NIfTI format, we still have to do some minor pre-processing before feeding the data to our model. 
        
        Generate sub-volumes:

        We are going to first generate "patches" of our data which you can think of as sub-volumes of the whole MR images. 
        - The reason that we are generating patches is because a network that can process the entire volume at once will simply not fit inside our current environment's memory/GPU.
        - Therefore we will be using this common technique to generate spatially consistent sub-volumes of our data, which can be fed into our network.
        - Specifically, we will be generating randomly sampled sub-volumes of shape [160, 160, 16] from our images. 
        - Furthermore, given that a large portion of the MRI volumes are just brain tissue or black background without any tumors, we want to make sure that we pick patches that at least include some amount of tumor data. 
        - Therefore, we are only going to pick patches that have at most 95% non-tumor regions (so at least 5% tumor). 
        - We do this by filtering the volumes based on the values present in the background labels.

        Standardization (mean 0, stdev 1):

        Lastly, given that the values in MR images cover a very wide range, we will standardize the values to have a mean of zero and standard deviation of 1. 
        - This is a common technique in deep image processing since standardization makes it much easier for the network to learn.

        In order to facilitate the training on the large dataset:
        - We have pre-processed the entire dataset into patches and stored the patches in the [`h5py`](http://docs.h5py.org/en/stable/) format. 
        - We also wrote a custom Keras [`Sequence`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence) class which can be used as a `Generator` for the keras model to train on large datasets. 
        """
        print(f"\n=== {self._PrepareData.__name__} ===")
        image_path = "data/BraTS-Data/imagesTr/BRATS_001.nii.gz"
        image_obj = nib.load(image_path)
        print(f'Type of the image {type(image_obj)}')
        # Extract data as numpy ndarray
        image_data = image_obj.get_fdata()
        type(image_data)
        # Get the image shape and print it out
        height, width, depth, channels = image_data.shape
        print(f"The image object has the following dimensions: height: {height}, width:{width}, depth:{depth}, channels:{channels}")
        # Define the data path and load the data
        label_path = "data/BraTS-Data/labelsTr/BRATS_001.nii.gz"
        label_obj = nib.load(label_path)
        type(label_obj)
        # Extract data labels
        label_array = label_obj.get_fdata()
        type(label_array)
        # Extract and print out the shape of the labels data
        height, width, depth = label_array.shape
        print(f"Dimensions of labels data array height: {height}, width: {width}, depth: {depth}")
        print(f'With the unique values: {numpy.unique(label_array)}')
        print("""Corresponding to the following label categories: 
                0: for normal 
                1: for edema
                2: for non-enhancing tumor 
                3: for enhancing tumor""")
        with open(f"{self._path}/processed/config.json") as json_file:
            config = json.load(json_file)
        # Get generators for training and validation sets
        self._train_generator = VolumeDataGenerator(config["train"], f"{self._path}/processed/train/", batch_size=self._batch_size, dim=(self._input_shape[1], self._input_shape[2], self._input_shape[3]), verbose=0)
        self._valid_generator = VolumeDataGenerator(config["valid"], f"{self._path}/processed/valid/", batch_size=self._batch_size, dim=(self._input_shape[1], self._input_shape[2], self._input_shape[3]), verbose=0)

    def _get_sub_volume(self, image, label, 
                    orig_x = 240, orig_y = 240, orig_z = 155, 
                    output_x = 160, output_y = 160, output_z = 16,
                    num_classes = 4, max_tries = 1000, 
                    background_threshold=0.95):
        """
        Extract random sub-volume from original images.

        Args:
            image (numpy.array): original image, 
                of shape (orig_x, orig_y, orig_z, num_channels)
            label (numpy.array): original label. 
                labels coded using discrete values rather than
                a separate dimension, 
                so this is of shape (orig_x, orig_y, orig_z)
            orig_x (int): x_dim of input image
            orig_y (int): y_dim of input image
            orig_z (int): z_dim of input image
            output_x (int): desired x_dim of output
            output_y (int): desired y_dim of output
            output_z (int): desired z_dim of output
            num_classes (int): number of class labels
            max_tries (int): maximum trials to do when sampling
            background_threshold (float): limit on the fraction 
                of the sample which can be the background

        returns:
            X (numpy.array): sample of original image of dimension 
                (num_channels, output_x, output_y, output_z)
            y (numpy.array): labels which correspond to X, of dimension 
                (num_classes, output_x, output_y, output_z)
        """
        # Initialize features and labels with `None`
        X = None
        y = None
        tries = 0
        while tries < max_tries:
            # randomly sample sub-volume by sampling the corner voxel
            # hint: make sure to leave enough room for the output dimensions!
            # do not remove/delete the '0's
            start_x = rng.integers(orig_x - output_x + 1, size=1)[0]
            start_y = rng.integers(orig_y - output_y + 1, size=1)[0]
            start_z = rng.integers(orig_z - output_z + 1, size=1)[0]
            # extract relevant area of label
            y = label[start_x: start_x + output_x,
                    start_y: start_y + output_y,
                    start_z: start_z + output_z]
            #print(f"1: y: {y.shape}")
            # One-hot encode the categories.
            # This adds a 4th dimension, 'num_classes'
            # (output_x, output_y, output_z, num_classes)
            y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
            #print(f"2: y: {y.shape}")
            # compute the background ratio (this has been implemented for you)
            bgrd_ratio = numpy.sum(y[:, :, :, 0])/(output_x * output_y * output_z)

            # increment tries counter
            tries += 1
            # if background ratio is below the desired threshold,
            # use that sub-volume.
            # otherwise continue the loop and try another random sub-volume
            if bgrd_ratio < background_threshold:

                # make copy of the sub-volume
                X = numpy.copy(image[start_x: start_x + output_x,
                                start_y: start_y + output_y,
                                start_z: start_z + output_z, :])
                # change dimension of X
                # from (x_dim, y_dim, z_dim, num_channels)
                # to (num_channels, x_dim, y_dim, z_dim)
                X = numpy.moveaxis(X, -1, 0)
                # change dimension of y
                # from (x_dim, y_dim, z_dim, num_classes)
                # to (num_classes, x_dim, y_dim, z_dim)
                y = numpy.moveaxis(y, -1, 0)
                
                # take a subset of y that excludes the background class
                # in the 'num_classes' dimension
                y = y[1:, :, :, :]
                return X, y
        # if we've tried max_tries number of samples
        # Give up in order to avoid looping forever.
        print(f"Tried {tries} times to find a sub-volume. Giving up...")

    def _standardize(self, image):
        """
        Standardize mean and standard deviation 
            of each channel and z_dimension.

        Args:
            image (numpy.array): input image, 
                shape (num_channels, dim_x, dim_y, dim_z)

        Returns:
            standardized_image (numpy.array): standardized version of input image
        """
        # initialize to array of zeros, with same shape as the image
        standardized_image = numpy.zeros_like(image)

        # iterate over channels
        for c in range(image.shape[0]):
            # iterate over the `z` dimension
            for z in range(image.shape[3]):
                # get a slice of the image 
                # at channel c and z-th dimension `z`
                image_slice = image[c,:,:,z]

                # subtract the mean from image_slice
                centered = image_slice - numpy.mean(image_slice)
                
                # divide by the standard deviation (only if it is different from zero)
                if numpy.std(centered) != 0:
                    centered_scaled = centered / numpy.std(centered)

                    # update  the slice of standardized image
                    # with the scaled centered and scaled image
                standardized_image[c, :, :, z] = centered_scaled if numpy.std(centered) != 0 else centered
        return standardized_image
    
if __name__ == "__main__":
    img_height = 160
    img_width = 160
    img_length = 16
    num_channels = 4
    EPOCHS = 1000
    BUFFER_SIZE = 500
    BATCH_SIZE = 3
    CLASSES = 3
    FILTERS = 32
    InitializeGPU()
    SetMemoryLimit(4096)
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='ImageSegmentation UNet')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    unet = ImageSegmentation3DUNet("data/BraTS-Data", "models/ImageSegmentation3DUNet.keras", (num_channels, img_height, img_width, img_length), FILTERS, CLASSES, BUFFER_SIZE, BATCH_SIZE, 0.001) # This is the default learning_rate value. The model accuracy drops drastically when I used 0.01
    unet.BuildTrainModel(EPOCHS, 20, 20, args.retrain)
    unet.show_predictions()