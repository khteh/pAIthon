from pathlib import Path
import glob, imageio, matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
from utils.GAN import restore_latest_checkpoint, TrainStep, Train, save_images, show_image, CreateGIF
from tensorflow.keras import layers, losses, optimizers, regularizers
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
"""
https://realpython.com/generative-adversarial-networks/
What Are Generative Adversarial Networks?
Generative adversarial networks are machine learning systems that can learn to mimic a given distribution of data. They were first proposed in a 2014 NeurIPS paper by deep learning expert Ian Goodfellow and his colleagues.

GANs consist of two neural networks, one trained to generate data and the other trained to distinguish fake data from real data (hence the “adversarial” nature of the model). Although the idea of a structure to generate data isn’t new, when it comes to image and video generation, GANs have provided impressive results such as:

Style transfer using CycleGAN, which can perform a number of convincing style transformations on images
Generation of human faces with StyleGAN, as demonstrated on the website This Person Does Not Exist
Structures that generate data, including GANs, are considered generative models in contrast to the more widely studied discriminative models.

TODO: WIP to convert from PyTorch used in the tutorial to use Tensorflow
"""
class Discriminator():
    model = None
    _cross_entropy = None
    optimizer = None
    def __init__(self):
        """
        Multilayer Perceptron NN defined in a sequential way using models.Sequential()
        The input is two-dimensional, and the first hidden layer is composed of 256 neurons with ReLU activation.
        The second and third hidden layers are composed of 128 and 64 neurons, respectively, with ReLU activation.
        The output is composed of a single neuron with sigmoidal activation to represent a probability.
        Pytorch:
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        https://stackoverflow.com/questions/66626700/difference-between-tensorflows-tf-keras-layers-dense-and-pytorchs-torch-nn-lin
        L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
        L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.
        """
        self.model = models.Sequential([
            layers.Dense(256, input_shape=(2,), activation='relu', name="L1", kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance.
            layers.Dropout(0.3),
            layers.Dense(128, input_shape=(256,), activation='relu', name="L2", kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(64, input_shape=(128,), activation='relu', name="L3", kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(1, input_shape=(64,), activation='linear', name="L4")]) # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
        """
        In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
        These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
        Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
        It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.
        More stable and accurate results can be obtained if the sigmoid/softmax and loss are combined during training.
        In the preferred organization the final layer has a linear activation. For historical reasons, the outputs in this form are referred to as *logits*. The loss function has an additional argument: `from_logits = True`. This informs the loss function that the sigmoid/softmax operation should be included in the loss calculation. This allows for an optimized implementation.
        logit = z. from_logits=True gives Tensorflow more flexibility in terms of how to compute this and whether or not it wnts to compyte g(z) explicitly. TensorFlow will compute z as an intermediate value, but it can rearrange terms to make this become computed more accurately with a little but less numerical roundoff error.
        """
        self._cross_entropy = losses.BinaryCrossentropy(from_logits=True) # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) 
        self.optimizer = optimizers.Adam(1e-4)

    def run(self, input, training: bool):
        return self.model(input, training=training)

    def loss(self, real, fake):
        """
        This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
        """
        real_loss = self._cross_entropy(tf.ones_like(real), real)
        fake_loss = self._cross_entropy(tf.zeros_like(fake), fake)
        return real_loss + fake_loss
    
    def UpdateParameters(self, tape, loss):
        # Use the gradient tape to automatically retrieve the gradients of the loss with respect to the trainable variables, dJ/dw.
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Run one step of gradient descent by updating the value of the variable to minimize the loss
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
   
class Generator():
    model = None
    _cross_entropy = None
    optimizer = None

    def __init__(self):
        """
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
        L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.
        """
        self.model = models.Sequential([
            layers.Input(shape=(100,)),
            layers.Dense(16, input_shape=(2,), activation='relu', name="L1", kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance.
            layers.Dense(32, input_shape=(16,), activation='relu', name="L2", kernel_regularizer=regularizers.l2(0.01)),
            # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
            layers.Dense(2, input_shape=(32,), name="L3")]) # Linear activation ("pass-through") if not specified
        """
        In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
        These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
        Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
        It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.
        More stable and accurate results can be obtained if the sigmoid/softmax and loss are combined during training.
        In the preferred organization the final layer has a linear activation. For historical reasons, the outputs in this form are referred to as *logits*. The loss function has an additional argument: `from_logits = True`. This informs the loss function that the sigmoid/softmax operation should be included in the loss calculation. This allows for an optimized implementation.
        logit = z. from_logits=True gives Tensorflow more flexibility in terms of how to compute this and whether or not it wnts to compyte g(z) explicitly. TensorFlow will compute z as an intermediate value, but it can rearrange terms to make this become computed more accurately with a little but less numerical roundoff error.
        """
        self._cross_entropy = losses.BinaryCrossentropy(from_logits=True) # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) 
        self.optimizer = optimizers.Adam(1e-4)

    def run(self, input, training: bool):
        return self.model(input, training = training)

    def loss(self, real, fake):
        """
        This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
        """
        real_loss = self._cross_entropy(tf.ones_like(real), real)
        fake_loss = self._cross_entropy(tf.zeros_like(fake), fake)
        return real_loss + fake_loss
    
    def UpdateParameters(self, tape, loss):
        # Use the gradient tape to automatically retrieve the gradients of the loss with respect to the trainable variables, dJ/dw.
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Run one step of gradient descent by updating the value of the variable to minimize the loss
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
def PrepareTrainingData(size: int, buffer_size: int, batch_size: int):
    """
    The training data is composed of pairs (x₁, x₂) so that x₂ consists of the value of the sine of x₁ for x₁ in the interval from 0 to 2π.
    PyTorch’ll need a tensor of labels, which are required by PyTorch’s data loader. Since GANs make use of unsupervised learning techniques, the labels can be anything. They won’t be used, after all.
    """
    data = numpy.zeros((size, 2))
    data[:,0] = 2 * math.pi * rng.random(size)
    data[:,1] = numpy.sin(data[:,0])
    data = data.reshape(data.shape[0], 2, 1).astype('float32')
    #data = tf.convert_to_tensor(data, dtype=tf.float64)
    print(f"data type: {type(data)}, shape: {data.shape}")
    #plt.plot(data[:, 0], data[:, 1], ".")
    #plt.show()
    """
    """
    #labels = tf.zeros((size, 1), dtype=tf.float64)
    #train = [(data[i], labels[i]) for i in range(size)] # [(array([ 4.74669199, -0.99941171]), np.float64(0.0)), ...]
    #print(f"train: {train[:10]}")
    #print(f"train data type: {type(train)}, shape: {train.shape}")
    return tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size).batch(batch_size)
    #return train

if __name__ == "__main__":
    SIZE = 1024
    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    EPOCHS = 300
    discriminator = Discriminator()
    generator = Generator()
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "sinewave_gan")
    train_dataset = PrepareTrainingData(SIZE, BUFFER_SIZE, BATCH_SIZE)
    checkpoint = Train(train_dataset, EPOCHS, discriminator, generator, checkpoint_prefix, BATCH_SIZE, 1, 1, 1)
    show_image(EPOCHS)
    gif = os.path.join(checkpoint_dir, "mnist_gan.gif")
    CreateGIF(gif)