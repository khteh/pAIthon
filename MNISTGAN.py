from pathlib import Path
import numpy, math, tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import numpy.lib.recfunctions as reconcile
import matplotlib.pyplot as plt
from utils.TensorModelPlot import PlotModelHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
# https://realpython.com/generative-adversarial-networks/

class Discriminator():
    _model = None
    def __init__(self):
        """
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )        
        """
        self._model = models.Sequential()
        self._model.add(layers.Dense(1024, input_shape=(784,), activation='relu'))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(512, input_shape=(1024,), activation='relu'))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(256, input_shape=(512,), activation='relu'))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(1, input_shape=(256,), activation='sigmoid'))
        self._model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
                metrics=['accuracy'])
    def forward(self, input):
        """
        To input the image coefficients into the MLP neural network, you vectorize them so that the neural network receives vectors with 784 coefficients.
        The call to x.view() converts the shape of the input tensor. In this case, the original shape of the input x is 32 × 1 × 28 × 28, where 32 is the batch size you’ve set up. 
        After the conversion, the shape of x becomes 32 × 784, with each line representing the coefficients of an image of the training set.
        """
        input = input.view(input.size(0), 784)
        output = self._model(input)
        return output

class Generator():
    _model = None
    def __init__(self):
        """
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
        """
        self._model = models.Sequential()
        self._model.add(layers.Dense(256, input_shape=(100,), activation='relu'))
        self._model.add(layers.Dense(512, input_shape=(256,), activation='relu'))
        self._model.add(layers.Dense(1024, input_shape=(512,), activation='relu'))
        # use the hyperbolic tangent function Tanh() as the activation of the output layer since the output coefficients should be in the interval from -1 to 1
        self._model.add(layers.Dense(784, input_shape=(1024,), activation='tanh'))

    def forward(self, input):
        output = self._model(input)
        output = output.view(input.size(0), 1, 28, 28)
        return output
    
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def PrepareMNISTData():
    """
    https://www.tensorflow.org/datasets/keras_example
    Build a training pipeline
    Apply the following transformations:

    - tf.data.Dataset.map: TFDS provide images of type tf.uint8, while the model expects tf.float32. Therefore, you need to normalize images.
    - tf.data.Dataset.cache As you fit the dataset in memory, cache it before shuffling for a better performance.
      Note: Random transformations should be applied after caching.
    - tf.data.Dataset.shuffle: For true randomness, set the shuffle buffer to the full dataset size.
      Note: For large datasets that can't fit in memory, use buffer_size=1000 if your system allows it.
    - tf.data.Dataset.batch: Batch elements of the dataset after shuffling to get unique batches at each epoch.
    - tf.data.Dataset.prefetch: It is good practice to end the pipeline by prefetching for performance.    
    """
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
