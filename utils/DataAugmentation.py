import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Resizing, Rescaling, RandomFlip, RandomRotation, RandomTranslation, RandomZoom
# https://www.tensorflow.org/tutorials/images/data_augmentation

def ResizeRescale(ds, size: int):
    resize_rescale = Sequential([
        Resizing(size, size),
        Rescaling(1./255)
    ])
    return ds.map(lambda x, y: (resize_rescale(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    datagen = ImageDataGenerator(
        rotation_range=40, #
        width_shift_range=0.2, #
        height_shift_range=0.2, #
        shear_range=0.2, # deprecated
        zoom_range=0.2, #
        horizontal_flip=True, #
        fill_mode='nearest' #)
    '''
    return Sequential([RandomFlip('horizontal'),
                       RandomRotation(0.2),
                       RandomTranslation(0.2, 0.2),
                       RandomZoom(0.2, 0.2)
                    ], name = "data_augmentation")

def AugmentData(ds, augment=False):
    # Use data augmentation only on the training set.
    if augment:
        augmenter = data_augmenter()
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)