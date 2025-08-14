import cv2, os, sys
import numpy as np
from pathlib import Path
import tensorflow as tf
from utils.GPU import InitializeGPU
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, optimizers, regularizers
"""
https://www.tensorflow.org/install/pip
$ pipenv run python -m traffic gtsrb/
$ pipenv run check50 --local ai50/projects/2024/x/traffic
"""
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main(model_path):
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
    if model_path and len(model_path) and Path(model_path).exists() and Path(model_path).is_file():
        print(f"Using saved model {model_path}...")
        model = tf.keras.models.load_model(model_path)
    else:
        # Get a compiled neural network
        model = get_model()
        # Fit model on training data

        # Epochs and batches
        # In the fit statement above, the number of epochs was set to 10. This specifies that the entire data set should be applied during training 10 times. During training, you see output describing the progress of training that looks like this:
        # Epoch 1/10
        # 6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
        # The first line, Epoch 1/10, describes which epoch the model is currently running. For efficiency, the training data set is broken into 'batches'. The default size of a batch in Tensorflow is 32. 
        # So, for example, if there are 200000 examples in our data set, there will be 6250 batches. The notation on the 2nd line 6250/6250 [==== is describing which batch has been executed.
        # Or, epochs = how many steps of a learning algorithm like gradient descent to run
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

    # Predict using linear activation with from_logits=True
    # This produces linear regression output (z). NOT g(z).
    #logits = model.predict(x_test)
    #f_x = tf.nn.softmax(logits) # g(z)

    # Save model to file
    if saveModel:
        #filename = sys.argv[2]
        model.save(model_path)
        print(f"Model saved to {model_path}.")

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

    In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
    These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
    Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
    It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.
    More stable and accurate results can be obtained if the sigmoid/softmax and loss are combined during training.
    In the preferred organization the final layer has a linear activation. For historical reasons, the outputs in this form are referred to as *logits*. The loss function has an additional argument: `from_logits = True`. This informs the loss function that the sigmoid/softmax operation should be included in the loss calculation. This allows for an optimized implementation.
    logit = z. from_logits=True gives Tensorflow more flexibility in terms of how to compute this and whether or not it wnts to compyte g(z) explicitly. TensorFlow will compute z as an intermediate value, but it can rearrange terms to make this become computed more accurately with a little but less numerical roundoff error.

    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.
    """
    model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='softmax', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='softmax', name="L1", kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance.
    layers.Dropout(0.5),
    layers.Dense(64, activation='softmax', name="L2", kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
    layers.Dense(NUM_CATEGORIES, name="L3")]) # Linear activation ("pass-through") if not specified
    """
    SparseCategorialCrossentropy or CategoricalCrossEntropy
    Tensorflow has two potential formats for target values and the selection of the loss defines which is expected.

    SparseCategorialCrossentropy: expects the target to be an integer corresponding to the index. For example, if there are 10 potential target values, y would be between 0 and 9.
    CategoricalCrossEntropy: Expects the target value of an example to be one-hot encoded where the value at the target index is 1 while the other N-1 entries are zero. An example with 10 potential target values, where the target is 2 would be [0,0,1,0,0,0,0,0,0,0].    
    """
    model.compile(optimizer='adam', # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
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
    main("models/traffic_lights.keras")
