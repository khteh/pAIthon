import matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Flatten, Dense, BatchNormalization, Reshape, Conv2DTranspose
from tensorflow.keras import losses, optimizers, regularizers
from utils.Image import ShowImage, CreateGIF
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotGANLossHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
# https://www.tensorflow.org/tutorials/generative/dcgan
# https://jonathan-hui.medium.com/gan-how-to-measure-gan-performance-64b988c47732
class Discriminator():
    """
    The discriminator is a CNN-based image classifier. It classifies the generated images as real or fake. The model will be trained to output positive values for real images, and negative values for fake images.
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection by "turning off" less important features or nodes in the network.
                               Useful when there are many features and some might be irrelevant, as it can effectively perform feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero. This helps to prevent individual weights from becoming excessively large and dominating the model.
                               Generally preferred in deep learning for its ability to smoothly reduce weight magnitudes and improve model generalization without completely removing features.
    """
    model = None
    _cross_entropy = None
    optimizer = None
    def __init__(self):
        self.model = Sequential([
                        Input(shape=(28, 28, 1)),
                        Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
                        # Input: 28x28x1
                        # Filter: 64 5x5 s:2 p = (f-1)/2 = 2
                        # Nh[l] = floor((Nh[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(28 + 4 - 5)/2 + 1 = 14
                        # Nw[l] = floor((Nw[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(28 + 4 - 5)/2 + 1 = 14
                        # Output volume = Nh[l] x Nw[l] x Nc[l] = 14 * 14 * 64 = 12544
                        LeakyReLU(),
                        Dropout(0.3),

                        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
                        # Input: 14 * 14 * 64
                        # Filter: 128 5x5 s:2 p = (f-1)/2 = 2
                        # Nh[l] = floor((Nh[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(14 + 4 - 5)/2 + 1 = 7
                        # Nw[l] = floor((Nw[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(14 + 4 - 5)/2 + 1 = 7
                        # Output volume = Nh[l] x Nw[l] x Nc[l] = 7 * 7 * 128 = 6272
                        LeakyReLU(),
                        Dropout(0.3),

                        Flatten(), # transforms the shape of the data from a n-dimensional array to a one-dimensional array.
                        # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
                        Dense(1, kernel_regularizer=regularizers.l2(0.01))]) # Linear activation ("pass-through") if not specified. Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected
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

    def forward(self, input, training: bool):
        """
        Function for completing a forward pass of the discriminator: Given an image tensor, returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        """
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
    The generator uses tf.keras.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). Start with a Dense layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. 
    Notice the tf.keras.LeakyReLU activation for each layer, except the output layer which uses tanh since the output coefficients should be in the interval from -1 to 1
    L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection by "turning off" less important features or nodes in the network.
                               Useful when there are many features and some might be irrelevant, as it can effectively perform feature selection.
    L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero. This helps to prevent individual weights from becoming excessively large and dominating the model.
                               Generally preferred in deep learning for its ability to smoothly reduce weight magnitudes and improve model generalization without completely removing features.
    Note: The tuorial code turns off bias use_bias=False only for the generator network. However, I don't see
    """
    model = None
    _cross_entropy = None
    optimizer = None
    def __init__(self):
        self.model = Sequential()
        self.model.add(Input(shape=(100,)))
        self.model.add(Dense(7*7*256, name="L1", kernel_regularizer=regularizers.l2(0.01))) # Decrease to fix high bias; Increase to fix high variance.
        self.model.add(BatchNormalization()) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        self.model.add(LeakyReLU())
        self.model.add(Reshape((7, 7, 256))) # Match the Dense layer's shape
        assert self.model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        self.model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
        # Input: 7x7x256
        # Filter: 128 5x5 s:1 p = (f-1)/2 = 2
        # Nh[l] = floor((Nh[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(7 + 4 - 5)/1 + 1 = 7
        # Nw[l] = floor((Nw[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(7 + 4 - 5)/1 + 1 = 7
        # Output volume = Nh[l] x Nw[l] x Nc[l] = 7 * 7 * 128 = 6272
        # Nout = (Nin  - 1) * s + f - 2p = (7-1) * 1 + 5 - 2*2 = 6 + 5 - 4 = 7
        assert self.model.output_shape == (None, 7, 7, 128)
        self.model.add(BatchNormalization()) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        self.model.add(LeakyReLU())

        self.model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
        # Input: 7x7x128
        # Filter: 64 5x5 s:2 p = (f-1)/2 = 2
        # Nh[l] = floor((Nh[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(7 + 4 - 5)/2 + 1 = 4
        # Nw[l] = floor((Nw[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(7 + 4 - 5)/2 + 1 = 4
        # Output volume = Nh[l] x Nw[l] x Nc[l] = 4 * 4 * 64 = 6272
        # Nout = (Nin  - 1) * s + f - 2p = (7-1) * 2 + 5 - 2*2 = 6*2 + 5 - 4 = 12 + 5 - 4 = 13
        # OutputSize = (InputSize - 1) * StrideSize + M + KernelSize - 2*P
        # M = (OutputSize - KernelSize + 2 * P) % StrideSize
        assert self.model.output_shape == (None, 14, 14, 64) # XXX
        self.model.add(BatchNormalization()) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        self.model.add(LeakyReLU())

        self.model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh')) # Similar to sigmoid graph but the output is [-1, 1]
        # Input: 14x14x64
        # Filter: 1 5x5 s:2 p = (f-1)/2 = 2
        # Nh[l] = floor((Nh[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(14 + 4 - 5)/2 + 1 = 7
        # Nw[l] = floor((Nw[l-1] + 2p[l] - f[l]) / s[l]) + 1 = floor(14 + 4 - 5)/2 + 1 = 7
        # Output volume = Nh[l] x Nw[l] x Nc[l] = 7 * 7 * 1 = 6272
        # Nout = (Nin  - 1) * s + f - 2p = (14-1) * 2 + 5 - 2*2 = 13*2 + 5 - 4 = 26 + 5 - 4 = 27
        assert self.model.output_shape == (None, 28, 28, 1) # XXX
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

    def forward(self, input, training: bool):
        """
        Function for completing a forward pass of the generator: Given a noise tensor, returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
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
    _name: str = None
    _buffer_size: int = None
    _batch_size: int = None
    _epochs: int = None
    _checkpoint_path: str = None
    _batch_dataset: tf.data.Dataset = None
    _generator: Generator = None
    _discriminator: Discriminator = None
    _checkpoint: tf.train.Checkpoint = None
    _noise_dim: int = None
    def __init__(self, name:str, buffer_size:int, batch_size:int, epochs:int, checkpoint_path: str):
        self._name = name
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
        self._PrepareMNISTData()
    def _PrepareMNISTData(self):
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
        #self._ShowMNISTImages(train_images)
        #print(f"train_images type: {type(train_images)}, shape: {train_images.shape}")
        #print(train_images[0])
        assert train_images.shape == (self._buffer_size, 28, 28)
        # https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg
        # Reshape the ndarray to (Height, Width, channels=1). 1-channel since it is a grayscale image.
        # This effectively transposes the pixel value (the last dimension) into a single column (28 rows)
        train_images = train_images.reshape(self._buffer_size, 28, 28, 1).astype('float32')
        assert train_images.shape == (self._buffer_size, 28, 28, 1)
        """
        The original tensors range from 0 to 1, and since the image backgrounds are black, most of the coefficients are equal to 0 when they’re represented using this range.
        Change the range of the coefficients to -1 to 1. With this transformation, the number of elements equal to 0 in the input samples is dramatically reduced, which helps in training the models.
        """
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        # Batch and shuffle the data
        self._batch_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self._buffer_size, reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE) # Batch dataset: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name=None). shape[0] = None is the batch size. None because it is flexible.
        #print(f"Batch dataset: {self._batch_dataset.element_spec}")

    def _ShowMNISTImages(self, images):
        fig, axes = plt.subplots(3,3,figsize=(5,5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[rng.choice(len(images))])
            ax.set_axis_off()
        plt.show()
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    # @tf.function decorator to increase performance. Without this decorator our training will take twice as long. If you would like to know more about how to increase performance with @tf.function take a look at the TensorFlow documentation.
    @tf.function
    def _TrainStep(self, images):
        """
        The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
        The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
        """
        #print(f"\n=== MNISTGAN._TrainStep ===")
        noise = rng.random([images.shape[0], self._noise_dim])
        #print(f"noise: {noise.shape}")
        # TensorFlow has the marvelous capability of calculating the derivatives for you. This is shown below. Within the tf.GradientTape() section, operations on Tensorflow Variables are tracked. When tape.gradient() is later called, it will return the gradient of the loss relative to the tracked variables. The gradients can then be applied to the parameters using an optimizer.
        # Tensorflow GradientTape records the steps used to compute cost J to enable auto differentiation.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self._generator.forward(noise, training=True)
            real_output = self._discriminator.forward(images, training=True)
            fake_output = self._discriminator.forward(generated_images, training=True)
            gen_loss = self._generator.loss(real_output, fake_output)
            disc_loss = self._discriminator.loss(real_output, fake_output)

        self._generator.UpdateParameters(gen_tape, gen_loss)
        # Use the GradientTape to calculate the gradients of the cost with respect to the parameter w: dJ/dw.
        self._discriminator.UpdateParameters(disc_tape, disc_loss)
        return gen_loss, disc_loss
    
    def Train(self, num_examples_to_generate: int, image_rows: int, image_cols: int):
        #num_examples_to_generate = 16
        # Reuse this seed overtime so that it's easier to visualize progress in the animated GIF
        gen_losses = []
        disc_losses = []
        seed = rng.random([num_examples_to_generate, self._noise_dim])
        #print(f"seed: {seed.shape}")
        """
        The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
        The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
        """
        for epoch in range(self._epochs):
            start = time.time()
            ave_gen_loss = 0
            ave_disc_loss = 0
            count = 0
            for image_batch in self._batch_dataset:
                gen_loss, disc_loss = self._TrainStep(image_batch)
                count += 1
                ave_gen_loss += gen_loss
                ave_disc_loss += disc_loss

            # Produce images for the GIF as you go
            self._save_images(self._generator.forward(seed, training=False), f"Generated Image at Epoch {epoch}", f'{self._name}_epoch_{epoch+1:04d}.png', (image_rows, image_cols))

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self._checkpoint.save(file_prefix = self._checkpoint_path)

            ave_gen_loss /= count
            ave_disc_loss /= count
            gen_losses.append(ave_gen_loss)
            disc_losses.append(ave_disc_loss)
            print(f"Epoch {epoch + 1} : {time.time()-start}s Generator Loss: {ave_gen_loss} Discriminator Loss: {ave_disc_loss}")

        # Generate after the final epoch
        PlotGANLossHistory("MNIST GAN", gen_losses, disc_losses)
        self._save_images(self._generator.forward(seed, training=False), f"Generated Image at Epoch {self._epochs}", f'{self._name}_epoch_{self._epochs:04d}.png', (image_rows, image_cols))

    def _save_images(self, data, title:str, filename: str, dimension):
        # Notice `training` is set to False. This is so all layers forward in inference mode (batchnorm).
        #fig = plt.figure(figsize=(4, 4))
        fig = plt.figure(figsize=dimension)
        #print(f"_save_images data.shape: {data.shape}, ndim: {data.ndim}") # data.shape: (16, 28, 28, 1), ndim: 4
        for i in range(data.shape[0]): # data.shape: (16, 28, 28, 1), ndim: 4
            plt.subplot(4, 4, i+1)
            plt.imshow(data[i, :, :, 0] * 127.5 + 127.5, cmap='gray') # The generator output shape is (, 28, 28, 1)
            plt.axis('off')
        #plt.legend()
        plt.suptitle(title)
        plt.savefig(f"output/MNISTGAN/{filename}")
        #plt.show()
        plt.close()
    def restore_latest_checkpoint(self):
        """
        https://www.tensorflow.org/tutorials/generative/dcgan
        """
        self._checkpoint.restore(tf.train.latest_checkpoint(self._checkpoint_path))

if __name__ == "__main__":
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 100
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "mnist_gan")
    Path("output/MNISTGAN").mkdir(parents=True, exist_ok=True)
    Path("output/MNISTGAN").is_dir()
    InitializeGPU()
    mnistGAN = MNISTGAN("mnist_gan", BUFFER_SIZE, BATCH_SIZE, EPOCHS, checkpoint_prefix)
    mnistGAN.Train(16, 4, 4)
    ShowImage(f'output/MNISTGAN/mnist_gan_epoch_{EPOCHS:04d}.png')
    CreateGIF("output/MNISTGAN/mnist_gan.gif", 'output/MNISTGAN/mnist_gan_epoch_*.png')