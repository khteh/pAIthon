from pathlib import Path
import glob, imageio, matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
from utils.GAN import show_image, CreateGIF
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

This module tries to convert from PyTorch used in the tutorial to use Tensorflow. However, this will NOT work because it needs dataset of sine wave images to train the Discriminator like how it works in MNISTGAN.py. So, the work in this module is ABORTED.
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
            layers.Input(shape=(1,2)),
            layers.Dense(256, activation='relu', name="L1", kernel_regularizer=regularizers.l2(0.01)), # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', name="L2", kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', name="L3", kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(1, activation='linear', name="L4")]) # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
        """
        In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
        These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
        Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
        It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.
        More stable and accurate results can be obtained if the sigmoid/softmax and loss are combined during training.
        In the preferred organization the final layer has a linear activation. For historical reasons, the outputs in this form are referred to as *logits*. The loss function has an additional argument: `from_logits = True`. This informs the loss function that the sigmoid/softmax operation should be included in the loss calculation. This allows for an optimized implementation.
        logit = z. from_logits=True gives Tensorflow more flexibility in terms of how to compute this and whether or not it wants to compute g(z) explicitly. TensorFlow will compute z as an intermediate value, but it can rearrange terms to make this become computed more accurately with a little but less numerical roundoff error.
        """
        self._cross_entropy = losses.BinaryCrossentropy(from_logits=True) # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) 
        self.optimizer = optimizers.Adam(1e-4) # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.

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
        self.model = models.Sequential()
        self.model.add(layers.Input(shape=(100,)))
        self.model.add(layers.Dense(16, activation='relu', name="L1", kernel_regularizer=regularizers.l2(0.01), use_bias=False)) # Decrease to fix high bias; Increase to fix high variance.
        print(f"Generator L1 output shape: {self.model.output_shape}")
        self.model.add(layers.Dense(32, activation='relu', name="L2", kernel_regularizer=regularizers.l2(0.01)))
        print(f"Generator L2 output shape: {self.model.output_shape}")
        # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
        # The output will consist of a vector with two elements that can be any value ranging from negative infinity to infinity, which will represent (x̃₁, x̃₂).
        self.model.add(layers.Dense(2, name="L3")) # Linear activation ("pass-through") if not specified
        print(f"Generator L3 output shape: {self.model.output_shape}")
        #print(f"Generator output shape: {self.model.output_shape}")
        assert self.model.output_shape == (None, 2)
        """
        In TensorFlow Keras, the from_logits argument in cross-entropy loss functions determines how the input predictions are interpreted. When from_logits=True, the loss function expects raw, unscaled output values (logits) from the model's last layer. 
        These logits are then internally converted into probabilities using the sigmoid or softmax function before calculating the cross-entropy loss. Conversely, when from_logits=False, the loss function assumes that the input predictions are already probabilities, typically obtained by applying a sigmoid or softmax activation function in the model's output layer.
        Using from_logits=True can offer numerical stability and potentially improve training, as it avoids the repeated application of the sigmoid or softmax function, which can lead to precision errors. 
        It is crucial to match the from_logits setting with the model's output activation to ensure correct loss calculation and effective training.
        More stable and accurate results can be obtained if the sigmoid/softmax and loss are combined during training.
        In the preferred organization the final layer has a linear activation. For historical reasons, the outputs in this form are referred to as *logits*. The loss function has an additional argument: `from_logits = True`. This informs the loss function that the sigmoid/softmax operation should be included in the loss calculation. This allows for an optimized implementation.
        logit = z. from_logits=True gives Tensorflow more flexibility in terms of how to compute this and whether or not it wants to compute g(z) explicitly. TensorFlow will compute z as an intermediate value, but it can rearrange terms to make this become computed more accurately with a little but less numerical roundoff error.
        """
        self._cross_entropy = losses.BinaryCrossentropy(from_logits=True) # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) 
        self.optimizer = optimizers.Adam(1e-4) # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.

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

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function decorator to increase performance. Without this decorator our training will take twice as long. If you would like to know more about how to increase performance with @tf.function take a look at the TensorFlow documentation.
@tf.function
def TrainStep(images, discriminator, generator, batch_size: int):
    noise_dim = 100
    """
    The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
    The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
    """
    noise = tf.random.normal([batch_size, noise_dim])
    # TensorFlow has the marvelous capability of calculating the derivatives for you. This is shown below. Within the tf.GradientTape() section, operations on Tensorflow Variables are tracked. When tape.gradient() is later called, it will return the gradient of the loss relative to the tracked variables. The gradients can then be applied to the parameters using an optimizer.
    # Tensorflow GradientTape records the steps used to compute cost J to enable auto differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator.run(noise, training=True)

      real_output = discriminator.run(images, training=True)
      fake_output = discriminator.run(generated_images, training=True)

      gen_loss = generator.loss(real_output, fake_output)
      disc_loss = discriminator.loss(real_output, fake_output)

    generator.UpdateParameters(gen_tape, gen_loss)
    # Use the GradientTape to calculate the gradients of the cost with respect to the parameter w: dJ/dw.
    discriminator.UpdateParameters(disc_tape, disc_loss)

def Train(dataset, epochs: int, discriminator, generator, checkpoint_path, batch_size: int, num_examples_to_generate: int, image_rows: int, image_cols: int):
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator.optimizer,
                                    discriminator_optimizer = discriminator.optimizer,
                                    generator = generator.model,
                                    discriminator = discriminator.model)
    noise_dim = 100
    #num_examples_to_generate = 16
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
        save_images(generator.run(seed, training=False), f"Generated Image at Epoch {epoch}", f'sinewave_gan_epoch_{epoch+1:04d}.png', (image_rows, image_cols))

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_path)

        print(f"Time for epoch {epoch + 1} is {time.time()-start}s")

    # Generate after the final epoch
    save_images(generator.run(seed, training=False), f"Generated Image at Epoch {epochs}", f'sinewave_gan_epoch_{epochs:04d}.png', (image_rows, image_cols))
    return checkpoint

def save_images(data, title:str, filename: str, dimension):
    # Notice `training` is set to False. This is so all layers run in inference mode (batchnorm).
    #fig = plt.figure(figsize=(4, 4))
    print(f"save_images data.shape: {data.shape}, ndim: {data.ndim}")
    fig, ax = plt.subplots(1,1)
    #plt.imshow(data[:, :]) # data.shape: (1, 2), ndim: 2 XXX: Should be (1024, 2)
    plt.plot(data[:, 0], data[:, 1], ".")
    plt.axis('off')
    #plt.legend()
    plt.suptitle(title)
    plt.savefig(f"output/{filename}")
    #plt.show()
    plt.close()

def PrepareTrainingData(buffer_size: int, batch_size: int):
    """
    The training data is composed of pairs (x₁, x₂) so that x₂ consists of the value of the sine of x₁ for x₁ in the interval from 0 to 2π.
    PyTorch’ll need a tensor of labels, which are required by PyTorch’s data loader. Since GANs make use of unsupervised learning techniques, the labels can be anything. They won’t be used, after all.
    """
    data = numpy.zeros((buffer_size, 2))
    data[:,0] = 2 * math.pi * rng.random(buffer_size)
    data[:,1] = numpy.sin(data[:,0])
    data = data.reshape(buffer_size, 1, 2).astype('float32') # This effectively transposes the pixel value (the last dimension) into a single column (28 rows)
    assert data.shape == (buffer_size, 1, 2)
    print(f"data type: {type(data)}, shape: {data.shape}") # data type: <class 'numpy.ndarray'>, shape: (1024, 2)
    return tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size).batch(batch_size)

if __name__ == "__main__":
    BUFFER_SIZE = 1024
    BATCH_SIZE = 32
    EPOCHS = 300
    discriminator = Discriminator()
    generator = Generator()
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "sinewave_gan")
    train_dataset = PrepareTrainingData(BUFFER_SIZE, BATCH_SIZE)
    print(f"train_dataset: {train_dataset.element_spec}") # train_dataset: TensorSpec(shape=(None, 2), dtype=tf.float64, name=None)
    Train(train_dataset, EPOCHS, discriminator, generator, checkpoint_prefix, BATCH_SIZE, 1, 1, 1)
    show_image(f'output/sinewave_gan_epoch_{EPOCHS:04d}.png')
    CreateGIF("output/sinewave_gan.gif", 'output/sinewave_gan_epoch_*.png')