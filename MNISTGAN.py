from pathlib import Path
import glob, imageio, matplotlib.pyplot as plt, os, PIL, time
from tensorflow.keras import layers, losses, optimizers
import numpy, math, tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import numpy.lib.recfunctions as reconcile
import matplotlib.pyplot as plt
from utils.TensorModelPlot import PlotModelHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
"""
https://realpython.com/generative-adversarial-networks/
https://www.tensorflow.org/tutorials/generative/dcgan
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
        The Discriminator
        The discriminator is a CNN-based image classifier.
        """
        self._model = models.Sequential()
        self._model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        self._model.add(layers.LeakyReLU())
        self._model.add(layers.Dropout(0.3))

        self._model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self._model.add(layers.LeakyReLU())
        self._model.add(layers.Dropout(0.3))

        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(1))
        """
        In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
        These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
        Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
        It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.    
        """
        self._cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        self._optimizer = optimizers.Adam(1e-4)

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
        The Generator
        The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). Start with a Dense layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. 
        Notice the tf.keras.layers.LeakyReLU activation for each layer, except the output layer which uses tanh since the output coefficients should be in the interval from -1 to 1
        """
        self._model = models.Sequential()
        self._model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        self._model.add(layers.BatchNormalization())
        self._model.add(layers.LeakyReLU())

        self._model.add(layers.Reshape((7, 7, 256)))
        assert self._model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        self._model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self._model.output_shape == (None, 7, 7, 128)
        self._model.add(layers.BatchNormalization())
        self._model.add(layers.LeakyReLU())

        self._model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self._model.output_shape == (None, 14, 14, 64)
        self._model.add(layers.BatchNormalization())
        self._model.add(layers.LeakyReLU())

        self._model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self._model.output_shape == (None, 28, 28, 1)
        """
        In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
        These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
        Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
        It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.    
        """
        self._cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        self._optimizer = optimizers.Adam(1e-4)

    def run(self, input, training: bool):
        return self._model(input, training)

    def loss(self, fake):
        """
        The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, compare the discriminators decisions on the generated images to an array of 1s.
        """
        return self._cross_entropy(tf.ones_like(fake), fake)

    def UpdateParameters(self, tape, loss):
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

def PrepareMNISTData(buffer_size: int, batch_size: int):
    """
    https://www.tensorflow.org/tutorials/generative/dcgan
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
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    """
    The original tensors range from 0 to 1, and since the image backgrounds are black, most of the coefficients are equal to 0 when they’re represented using this range.
    Change the range of the coefficients to -1 to 1. With this transformation, the number of elements equal to 0 in the input samples is dramatically reduced, which helps in training the models.
    """
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
    return train_dataset

def restore_latest_checkpoint(checkpoint, checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def TrainStep(images, discriminator, generator, batch_size: int):
    noise_dim = 100
    """
    The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
    The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
    """
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator.run(noise, training=True)

      real_output = discriminator.run(images, training=True)
      fake_output = discriminator.run(generated_images, training=True)

      gen_loss = generator.loss(fake_output)
      disc_loss = discriminator.loss(real_output, fake_output)

    generator.UpdateParameters(gen_tape, gen_loss)
    discriminator.UpdateParameters(disc_tape, disc_loss)

def Train(dataset, epochs: int, discriminator, generator, checkpoint_path, batch_size: int):
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator.optimizer,
                                    discriminator_optimizer = discriminator.optimizer,
                                    generator = generator,
                                    discriminator = discriminator)
    noise_dim = 100
    num_examples_to_generate = 16
    # Reuse this seed overtime so that it's easier to visualize progress in the animated GIF
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    """
    The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
    The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
    """
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            TrainStep(image_batch, discriminator, generator, batch_size)

        # Produce images for the GIF as you go
        save_images(generator.run(seed, training=False), epoch + 1)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_path)

        print(f"Time for epoch {epoch + 1} is {time.time()-start}s")

    # Generate after the final epoch
    save_images(generator.run(seed, training=False), epochs)
    return checkpoint

def save_images(data, epoch: int):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    fig = plt.figure(figsize=(4, 4))
    for i in range(data.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(data[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()

def show_image(epoch: int):
    # Display a single image using the epoch number
  return PIL.Image.open(f'image_at_epoch_{epoch:04d}.png')

def CreateGIF(filename: str):
    """
    Use imageio to create an animated gif using the images saved during training.
    """
    with imageio.get_writer(filename, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)
        image = imageio.imread(f)
        writer.append_data(image)

if __name__ == "__main__":
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 50
    discriminator = Discriminator()
    generator = Generator()   
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "mnist_gan")
    train_dataset = PrepareMNISTData(BUFFER_SIZE, BATCH_SIZE)
    checkpoint = Train(train_dataset, EPOCHS, discriminator, generator, checkpoint_prefix, BATCH_SIZE)
    show_image(EPOCHS)
    gif = os.path.join(checkpoint_dir, "mnist_gan.gif")
    CreateGIF(gif)