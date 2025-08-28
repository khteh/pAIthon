from pathlib import Path
import matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras import layers, losses, optimizers, regularizers
from utils.GAN import restore_latest_checkpoint, show_image, CreateGIF
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
class Discriminator():
    """
    The discriminator is a CNN-based image classifier. It classifies the generated images as real or fake. The model will be trained to output positive values for real images, and negative values for fake images.
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.
    """
    model = None
    _cross_entropy = None
    optimizer = None
    def __init__(self):
        self.model = models.Sequential([
                        layers.Input(shape=(28, 28, 1)),
                        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
                        layers.LeakyReLU(),
                        layers.Dropout(0.3),

                        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
                        layers.LeakyReLU(),
                        layers.Dropout(0.3),

                        layers.Flatten(), # transforms the shape of the data from a n-dimensional array to a one-dimensional array.
                        # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
                        layers.Dense(1, kernel_regularizer=regularizers.l2(0.01))]) # Linear activation ("pass-through") if not specified. Densely connected, or fully connected
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
    """
    The generator will generate handwritten digits resembling the MNIST data.
    The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). Start with a Dense layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. 
    Notice the tf.keras.layers.LeakyReLU activation for each layer, except the output layer which uses tanh since the output coefficients should be in the interval from -1 to 1
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero.      
    """
    model = None
    _cross_entropy = None
    optimizer = None
    def __init__(self):
        self.model = models.Sequential()
        self.model.add(layers.Input(shape=(100,)))
        self.model.add(layers.Dense(7*7*256, use_bias=False, name="L1", kernel_regularizer=regularizers.l2(0.01))) # Decrease to fix high bias; Increase to fix high variance.
        self.model.add(layers.BatchNormalization()) # stabilize the learning process, accelerate convergence, and potentially improve generalization performance.
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Reshape((7, 7, 256)))
        assert self.model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        self.model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 7, 7, 128)
        self.model.add(layers.BatchNormalization()) # stabilize the learning process, accelerate convergence, and potentially improve generalization performance.
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 14, 14, 64)
        self.model.add(layers.BatchNormalization()) # stabilize the learning process, accelerate convergence, and potentially improve generalization performance.
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.model.output_shape == (None, 28, 28, 1)
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
        """
        Upsamples the input.
        """
        return self.model(input, training=training)

    def loss(self, real, fake):
        """
        The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, compare the discriminators decisions on the generated images to an array of 1s.
        """
        return self._cross_entropy(tf.ones_like(fake), fake)

    def UpdateParameters(self, tape, loss):
        # Use the gradient tape to automatically retrieve the gradients of the loss with respect to the trainable variables, dJ/dw.
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Run one step of gradient descent by updating the value of the variable to minimize the loss
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

class MNISTGAN():
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
    _buffer_size: int = None
    _batch_size: int = None
    _epochs: int = None
    _checkpoint_path: str = None
    _batch_dataset: tf.data.Dataset = None
    _generator: Generator = None
    _discriminator: Discriminator = None
    _checkpoint: tf.train.Checkpoint = None
    _noise_dim: int = None
    def __init__(self, buffer_size:int, batch_size:int, epochs:int, checkpoint_path: str):
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._epochs = epochs
        self._noise_dim = 100
        self._checkpoint_path = checkpoint_path
        self._generator = Generator()
        self._discriminator = Discriminator()
        self._checkpoint = tf.train.Checkpoint(generator_optimizer = self._generator.optimizer,
                                        discriminator_optimizer = self._discriminator.optimizer,
                                        generator = self._generator.model,
                                        discriminator = self._discriminator.model)
    def PrepareMNISTData(self):
        """
        https://www.tensorflow.org/tutorials/generative/dcgan
        https://www.tensorflow.org/datasets/keras_example
        https://keras.io/api/datasets/mnist/
        This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
        Training Images (x_train):
            Shape: (60000, 28, 28)
            Dimensions: 3 dimensions (number of samples, height, width)
            Interpretation: 60,000 grayscale images, each 28 pixels by 28 pixels.

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
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data() # train_images type: <class 'numpy.ndarray'>, shape: (60000, 28, 28)
        #print(f"train_images type: {type(train_images)}")
        assert train_images.shape == (self._buffer_size, 28, 28)
        #print("train_images: ")
        #print(train_images[10])
        # https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg
        # Reshape the ndarray to (Height, Width, channels=1). 1-channel since it is a grayscale image.
        # This effectively transposes the pixel value (the last dimension) into a single column (28 rows)
        train_images = train_images.reshape(self._buffer_size, 28, 28, 1).astype('float32')
        assert train_images.shape == (self._buffer_size, 28, 28, 1)
        #print(f"train_images type: {type(train_images)}, shape: {train_images.shape}")
        #print(train_images[5])
        #print(train_images[3][5])
        """
        The original tensors range from 0 to 1, and since the image backgrounds are black, most of the coefficients are equal to 0 when they’re represented using this range.
        Change the range of the coefficients to -1 to 1. With this transformation, the number of elements equal to 0 in the input samples is dramatically reduced, which helps in training the models.
        """
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        # Batch and shuffle the data
        self._batch_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self._buffer_size).batch(self._batch_size) # Batch dataset: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name=None). shape[0] = None is the batch size. None because it is flexible.
        #print(f"Batch dataset: {self._batch_dataset.element_spec}")

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    # @tf.function decorator to increase performance. Without this decorator our training will take twice as long. If you would like to know more about how to increase performance with @tf.function take a look at the TensorFlow documentation.
    @tf.function
    def _TrainStep(self, images):
        """
        The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
        The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
        """
        noise = tf.random.normal([self._batch_size, self._noise_dim])
        # TensorFlow has the marvelous capability of calculating the derivatives for you. This is shown below. Within the tf.GradientTape() section, operations on Tensorflow Variables are tracked. When tape.gradient() is later called, it will return the gradient of the loss relative to the tracked variables. The gradients can then be applied to the parameters using an optimizer.
        # Tensorflow GradientTape records the steps used to compute cost J to enable auto differentiation.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self._generator.run(noise, training=True)

            real_output = self._discriminator.run(images, training=True)
            fake_output = self._discriminator.run(generated_images, training=True)

            gen_loss = self._generator.loss(real_output, fake_output)
            disc_loss = self._discriminator.loss(real_output, fake_output)

        self._generator.UpdateParameters(gen_tape, gen_loss)
        # Use the GradientTape to calculate the gradients of the cost with respect to the parameter w: dJ/dw.
        self._discriminator.UpdateParameters(disc_tape, disc_loss)

    def Train(self, num_examples_to_generate: int, image_rows: int, image_cols: int):
        #num_examples_to_generate = 16
        # Reuse this seed overtime so that it's easier to visualize progress in the animated GIF
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        """
        The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
        The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
        """
        for epoch in range(self._epochs):
            start = time.time()

            for image_batch in self._batch_dataset:
                self._TrainStep(image_batch)

            # Produce images for the GIF as you go
            self._save_images(self._generator.run(seed, training=False), f"Generated Image at Epoch {epoch}", f'mnist_gan_epoch_{epoch+1:04d}.png', (image_rows, image_cols))

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self._checkpoint.save(file_prefix = self._checkpoint_path)

            print(f"Time for epoch {epoch + 1} is {time.time()-start}s")

        # Generate after the final epoch
        self._save_images(self._generator.run(seed, training=False), f"Generated Image at Epoch {self._epochs}", f'mnist_gan_epoch_{self._epochs:04d}.png', (image_rows, image_cols))

    def _save_images(self, data, title:str, filename: str, dimension):
        # Notice `training` is set to False. This is so all layers run in inference mode (batchnorm).
        #fig = plt.figure(figsize=(4, 4))
        fig = plt.figure(figsize=dimension)
        #print(f"_save_images data.shape: {data.shape}, ndim: {data.ndim}")
        for i in range(data.shape[0]): # data.shape: (16, 28, 28, 1), ndim: 4
            plt.subplot(4, 4, i+1)
            plt.imshow(data[i, :, :, 0] * 127.5 + 127.5, cmap='gray') # The generator output shape is (, 28, 28, 1)
            plt.axis('off')
        #plt.legend()
        plt.suptitle(title)
        plt.savefig(f"output/{filename}")
        #plt.show()
        plt.close()

if __name__ == "__main__":
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 50
    noise_dim = 100
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    seed = tf.random.normal([16, noise_dim])
    print(f"noise: {noise.shape} ndim: {noise.ndim}, seed: {seed.shape} ndim: {seed.ndim}")
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "mnist_gan")
    mnistGAN = MNISTGAN(BUFFER_SIZE, BATCH_SIZE, EPOCHS, checkpoint_prefix)
    mnistGAN.PrepareMNISTData()
    mnistGAN.Train(16, 4, 4)
    show_image(f'output/mnist_gan_epoch_{EPOCHS:04d}.png')
    CreateGIF("output/mnist_gan.gif", 'output/mnist_gan_epoch_*.png')