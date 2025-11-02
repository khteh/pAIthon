import argparse, os, numpy, pandas as pd, imageio.v3 as iio
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, BatchNormalization, Normalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from utils.GPU import InitializeGPU, SetMemoryLimit
from utils.TrainingMetricsPlot import PlotModelHistory
from utils.TrainingUtils import CreateTensorBoardCallback, CreateCircuitBreakerCallback
"""
data too large to keep in git:
19M	CameraMask
507M	CameraRGB
"""
class ImageSegmentationUNet():
    _path:str = None
    _input_size = None
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
    def __init__(self, path:str, input_size, n_filters:int, n_classes: int, buffer_size:int, batch_size:int, learning_rate:float):
        self._path = path
        self._input_size = input_size
        self._n_filters = n_filters
        self._n_classes = n_classes
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._normalization = Normalization(axis=-1)
        self._circuit_breaker = CreateCircuitBreakerCallback("val_accuracy", "max", 5)
        self._PrepareData()
        if self._path and len(self._path) and Path(self._path).exists() and Path(self._path).is_file():
            print(f"Using saved model {self._path}...")
            self._model = tf.keras.models.load_model(self._path)

    def Encoder(self, inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
        """
        Convolutional downsampling block
        The encoder is a stack of various Encoders:

        Each Encoder() is composed of 2 Conv2D layers with ReLU activations. We will apply Dropout, and MaxPooling2D to some Encoders, as you will verify in the following sections, specifically to the last two blocks of the downsampling.

        The function will return two tensors:

        next_layer: That will go into the next block.
        skip_connection: That will go into the corresponding decoding block.
        Note: If max_pooling=True, the next_layer will be the output of the MaxPooling2D layer, but the skip_connection will be the output of the previously applied layer(Conv2D or Dropout, depending on the case). Else, both results will be identical.
        
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
        conv = BatchNormalization()(conv)
        conv = Conv2D(n_filters, # Number of filters
                    3,   # Kernel size
                    activation="relu",
                    padding="same",
                    # set 'kernel_initializer' same as above
                    kernel_initializer='he_normal')(conv)
        conv = BatchNormalization()(conv)
        
        # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
        if dropout_prob > 0:
            conv = Dropout(dropout_prob)(conv)
            
        # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
        next_layer = MaxPooling2D(2)(conv) if max_pooling else conv
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
        conv = BatchNormalization()(conv)
        conv = Conv2D(n_filters,  # Number of filters
                    3,   # Kernel size
                    activation="relu",
                    padding="same",
                    # set 'kernel_initializer' same as above
                    kernel_initializer='he_normal')(conv)
        conv = BatchNormalization()(conv)
        return conv

    def BuildTrainModel(self, epochs: int, retrain:bool = False):
        """
        Unet model
        
        Arguments:
            input_size -- Input shape 
            n_filters -- Number of filters for the convolutional layers
            n_classes -- Number of output classes
        Returns: 
            model -- tf.keras.Model
        """
        print(f"\n=== {self.BuildTrainModel.__name__} ===")
        new_model = not self._model
        if not self._model:
            inputs = Input(self._input_size)
            #print(f"inputs: {tf.math.reduce_mean(inputs, axis=-1)}")
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
            self._model.compile(optimizer=Adam(self._learning_rate),
                        loss=SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            self._model.summary()
            plot_model(
                self._model,
                to_file="output/ImageSegmentationUNetModel.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="LR", # rankdir argument passed to PyDot, a string specifying the format of the plot: "TB" creates a vertical plot; "LR" creates a horizontal plot.
                expand_nested=True,
                show_layer_activations=True)
        if new_model or retrain:
            tensorboard = CreateTensorBoardCallback("ImageSegmentationUNet") # Create a new folder with current timestamp
            history = self._model.fit(self._train_dataset, epochs=epochs, shuffle=True, validation_data = self._val_dataset, validation_freq=1, callbacks=[tensorboard, self._circuit_breaker]) # Use kwargs. Otherwise, the epochs will be treated as the labels and will hit error since this is using tf.data.Dataset which includes BOTH X and Y.
            PlotModelHistory("ImageSegmentation UNet", history)
            if self._path:
                self._model.save(self._path)
                print(f"Model saved to {self._path}.")

    def _display(self, images):
        titles = ['Input Image', 'True Mask', 'Predicted Mask']
        fig, axes = plt.subplots(len(images), len(titles), constrained_layout=True, figsize=(15, 20)) # figsize = (width, height)
        # rect=[0, 0, 1, 0.98] tells tight_layout to arrange the subplots within the bottom 98% of the figure's height, leaving the top 2% some space for the suptitle, for instance.
        fig.tight_layout(pad=0.1,rect=[0, 0, 1, 0.98]) #[left, bottom, right, top]
        for i in range(len(images)):
            for j in range(len(titles)):
                axes[i][j].title.set_text(titles[j])
                axes[i][j].imshow(tf.keras.preprocessing.image.array_to_img(images[i][j]))
                axes[i][j].axis('off')
        plt.savefig(f"output/ImageSegmentationUNet.png")
        #plt.show()
        plt.close()

    def _create_mask(self, pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]            

    def show_predictions(self, image, mask, num=1):
        """
        Displays the first image of each of the num batches.
        XXX: The predictions are all empty.
        """
        print(f"\n=== {self.show_predictions.__name__} ===")
        images = []
        if image and mask:
            images.append([image, mask, self._create_mask(self._model.predict(image[tf.newaxis, ...]))])
        else:
            for image, mask in self._train_dataset.take(num):
                pred_mask = self._model.predict(image)
                print(f"image: {image.shape}, mask: {mask.shape}, pred_mask: {type(pred_mask)} {pred_mask.shape} {numpy.mean(pred_mask)}")
                images.append([image[0], mask[0], self._create_mask(pred_mask)])
        self._display(images)

    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        image_path = './data/CameraRGB/'
        mask_path = './data/CameraMask/'
        image_list_orig = os.listdir(image_path)
        image_list = [image_path+i for i in image_list_orig]
        mask_list = [mask_path+i for i in image_list_orig]
        print(f"image_list: {len(image_list)}, maks_list: {len(mask_list)}") # 1060
        image_filenames = tf.constant(image_list)
        masks_filenames = tf.constant(mask_list)
        training_dataset = tf.data.Dataset.from_tensor_slices((image_filenames[:-100], masks_filenames[:-100]))
        validation_dataset = tf.data.Dataset.from_tensor_slices((image_filenames[-100:], masks_filenames[-100:]))
        training_image_ds = training_dataset.map(self._process_path)
        processed_training_image_ds = training_image_ds.map(self._preprocess)
        validation_image_ds = training_dataset.map(self._process_path)
        processed_validation_image_ds = training_image_ds.map(self._preprocess)
        print(f"processed_training_image_ds: {processed_training_image_ds.element_spec}, processed_validation_image_ds: {processed_validation_image_ds.element_spec}")
        self._train_dataset = processed_training_image_ds.shuffle(self._buffer_size, reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self._normalization.adapt(self._train_dataset.map(lambda x, y: x))
        self._val_dataset = processed_training_image_ds.shuffle(self._buffer_size, reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        #mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])
        """
        N = 2
        img = iio.imread(image_list[N])
        mask = iio.imread(mask_list[N])
        fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(14, 10)) # figsize = (width, height)
        fig.tight_layout(pad=0.1,rect=[0, 0, 1, 0.98]) #[left, bottom, right, top]
        axes[0].imshow(img)
        axes[0].set_title('Image')
        axes[1].imshow(mask[:, :, 0])
        axes[1].set_title('Segmentation')
        axes[0].grid(False)
        axes[1].grid(False)
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        plt.show()
        """

    def _process_path(self, image_path, mask_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
        return img, mask

    def _preprocess(self, image, mask):
        input_image = tf.image.resize(image, (96, 128), method='nearest')
        input_mask = tf.image.resize(mask, (96, 128), method='nearest')
        return input_image, input_mask

if __name__ == "__main__":
    img_height = 96
    img_width = 128
    num_channels = 3
    EPOCHS = 30
    BUFFER_SIZE = 500
    BATCH_SIZE = 32
    CLASSES = 23
    FILTERS = 32
    InitializeGPU()
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='ImageSegmentation UNet')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    unet = ImageSegmentationUNet("models/ImageSegmentationUNet.keras", (img_height, img_width, num_channels), FILTERS, CLASSES, BUFFER_SIZE, BATCH_SIZE, 0.001) # This is the default learning_rate value. The model accuracy drops drastically when I used 0.01
    unet.BuildTrainModel(EPOCHS, args.retrain)
    unet. show_predictions(None, None, 10)