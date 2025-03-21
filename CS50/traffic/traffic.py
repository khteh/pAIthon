import cv2
import numpy as np
import os
from pathlib import Path
import sys
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
"""
https://www.tensorflow.org/install/pip
$ python traffic.py gtsrb/
$ check50 --local ai50/projects/2024/x/traffic
"""
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

# Hide GPU from visible devices
def InitializeGPU():
    """
    2024-12-17 12:39:33.030218: I external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:1193] failed to allocate 2.2KiB (2304 bytes) from device: RESOURCE_EXHAUSTED: : CUDA_ERROR_OUT_OF_MEMORY: out of memory
    https://stackoverflow.com/questions/39465503/cuda-error-out-of-memory-in-tensorflow
    https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    https://www.tensorflow.org/guide/gpu
    """
    #tf.config.set_visible_devices([], 'GPU')
    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"{len(gpus)} GPUs available")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.keras]")
    print(f"TF v{tf.version.VERSION}")
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    saveModel: bool = False
    if (len(sys.argv) == 3) and sys.argv[2] and Path(sys.argv[2]).exists() and Path(sys.argv[2]).is_file():
        print(f"Using saved model {sys.argv[2]}...")
        model = tf.keras.models.load_model(sys.argv[2])
    else:
        # Get a compiled neural network
        model = get_model()
        # Fit model on training data
        history = model.fit(x_train, y_train, epochs=EPOCHS)
        saveModel = True

    # Display the model's architecture
    print("Model Summary:")
    model.summary()

    # Evaluate neural network performance
    train_loss, train_accuracy = model.evaluate(x_train,  y_train, verbose=2)
    test_loss, test_accuracy = model.evaluate(x_test,  y_test, verbose=2)
    print(f'Training accuracy: {train_accuracy:.4f}, loss: {train_loss:.4f}')
    print(f'Testing accuracy: {test_accuracy:.4f}, loss: {test_loss:.4f}')
    # Save model to file
    if saveModel and len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    if os.path.isdir(os.fsdecode(data_dir)):
        for p in os.listdir(data_dir):
            dir = os.path.join(data_dir, os.fsdecode(p))
            #print(f"Processing {dir}...")
            if os.path.isdir(dir):
                category = os.fsdecode(p)
                #print(f"category: {category}")
                for f in os.listdir(dir):
                    if f.endswith(".ppm"):
                        file = os.path.join(dir, f)
                        #print(f"file: {file}")
                        img = cv2.imread(file)   # reads an image in the BGR format
                        img = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
                        #img = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT))
                        assert img.shape == (IMG_WIDTH, IMG_HEIGHT, 3)
                        assert img.ndim == 3
                        labels.append(category)
                        images.append(img)
            else:
                raise Exception(f"data_dir {p} not a valid directory!")
    else:
        raise Exception(f"data_dir {data_dir} not a valid directory!")
    print(f"{len(images)} images; {len(labels)} labels")
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='softmax', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='softmax'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='softmax'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CATEGORIES))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
                metrics=['accuracy'])
    return model
"""
Only executes when this file is run as a script
Does NOT execute when this file is imported as a module
__name__ stores the name of a module when it is loaded. It is set to either the string of "__main__" if it's in the top-level or the module's name if it is being imported.
"""
if __name__ == "__main__":
    InitializeGPU()
    main()
