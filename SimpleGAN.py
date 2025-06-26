from pathlib import Path
import glob, imageio, matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import numpy.lib.recfunctions as reconcile
import matplotlib.pyplot as plt
from utiuls.GAN import restore_latest_checkpoint, TrainStep, Train, save_images, show_image, CreateGIF
from tensorflow.keras import layers, losses, optimizers
from utils.TensorModelPlot import PlotModelHistory
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
"""
class Discriminator():
    _model = None
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
        """
        self._model = models.Sequential()
        self._model.add(layers.Dense(256, input_shape=(2,), activation='relu', name="L1"))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(128, input_shape=(256,), activation='relu', name="L2"))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(64, input_shape=(128,), activation='relu', name="L3"))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(1, input_shape=(64,), activation='linear', name="L4")) # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
        """
        In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
        These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
        Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
        It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.
        logit = z. from_logits=True gives Tensorflow more flexibility in terms of how to compute this and whether or not it wnts to compyte g(z) explicitly. TensorFlow will compute z as an intermediate value, but it can rearrange terms to make this become computed more accurately with a little but less numerical roundoff error.
        """
        self._cross_entropy = losses.BinaryCrossentropy(from_logits=True) # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) 
        self.optimizer = optimizers.Adam(1e-4)

    def run(self, input, training: bool):
        return self._model(input, training)

    def loss(self, real, fake):
        """
        This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
        """
        real_loss = self._cross_entropy(tf.ones_like(real), real)
        fake_loss = self._cross_entropy(tf.zeros_like(fake), fake)
        return real_loss + fake_loss
    
    def UpdateParameters(self, tape, loss):
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
   
class Generator():
    _model = None
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
        """
        self._model = models.Sequential()
        self._model.add(layers.Dense(16, input_shape=(2,), activation='relu', name="L1"))
        self._model.add(layers.Dense(32, input_shape=(16,), activation='relu', name="L2"))
        # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
        self._model.add(layers.Dense(2, input_shape=(32,), name="L3")) # Linear activation ("pass-through") if not specified
        """
        In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
        These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
        Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
        It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.
        logit = z. from_logits=True gives Tensorflow more flexibility in terms of how to compute this and whether or not it wnts to compyte g(z) explicitly. TensorFlow will compute z as an intermediate value, but it can rearrange terms to make this become computed more accurately with a little but less numerical roundoff error.
        """
        self._cross_entropy = losses.BinaryCrossentropy(from_logits=True) # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) 
        self.optimizer = optimizers.Adam(1e-4)

    def run(self, input, training: bool):
        return self._model(input, training)

    def loss(self, real, fake):
        """
        This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
        """
        real_loss = self._cross_entropy(tf.ones_like(real), real)
        fake_loss = self._cross_entropy(tf.zeros_like(fake), fake)
        return real_loss + fake_loss
    
    def UpdateParameters(self, tape, loss):
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
    
def PrepareTrainingData(size: int, buffer_size: int, batch_size: int):
    data = numpy.zeroes((size, 2))
    data[:,0] = 2 * math.pi * rng.random(size)
    data[:,1] = math.sin(data[:,0])
    labels = numpy.zeroes(size)
    train = [(data[i], labels[i]) for i in range(size)]
    return tf.data.Dataset.from_tensor_slices(train).shuffle(buffer_size).batch(batch_size)

if __name__ == "__main__":
    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    EPOCHS = 300
    discriminator = Discriminator()
    generator = Generator()
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "sinewave_gan")
    train_dataset = PrepareTrainingData(BUFFER_SIZE, BATCH_SIZE)
    checkpoint = Train(train_dataset, EPOCHS, discriminator, generator, checkpoint_prefix, BATCH_SIZE, 1, 1, 1)
    show_image(EPOCHS)
    gif = os.path.join(checkpoint_dir, "mnist_gan.gif")
    CreateGIF(gif)