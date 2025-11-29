import glob, imageio, matplotlib.pyplot as plt, os, time
import numpy, math, tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from utils.Image import CreateGIF, ShowImage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Flatten, Dense, BatchNormalization, Reshape, Conv2DTranspose
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.regularizers import l2
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotGANLossHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
# https://jonathan-hui.medium.com/gan-how-to-measure-gan-performance-64b988c47732
class Discriminator():
    _samples: int = None
    model = None
    _cross_entropy = None
    optimizer = None
    def __init__(self, samples: int):
        """
        Multilayer Perceptron NN defined in a sequential way using Sequential()
        The input is two-dimensional, and the first hidden layer is composed of 256 neurons with ReLU activation.
        The second and third hidden layers are composed of 128 and 64 neurons, respectively, with ReLU activation.
        The output is composed of a single neuron with sigmoidal activation to represent a probability.

        L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection by "turning off" less important features or nodes in the network.
                                   Useful when there are many features and some might be irrelevant, as it can effectively perform feature selection.
        L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero. This helps to prevent individual weights from becoming excessively large and dominating the model.
                                   Generally preferred in deep learning for its ability to smoothly reduce weight magnitudes and improve model generalization without completely removing features.
        """
        self._samples = samples
        self.model = Sequential([
            Input(shape=(self._samples, 2)),
            Dense(256, activation='relu', name="L1", kernel_regularizer=l2(0.01)), # Decrease to fix high bias; Increase to fix high variance. Densely connected, or fully connected
            Dropout(0.3),
            Dense(128, activation='relu', name="L2", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(64, activation='relu', name="L3", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(1, activation='linear', name="L4")]) # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
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
    Takes samples from a latent space as its input and generates data resembling data in the training dataset.
    Use NN as multilayer perceptron.
    """
    _samples: int = None
    model = None
    _cross_entropy = None
    optimizer = None
    def __init__(self, samples: int):
        """
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        L1 Regularization (Lasso): Penalizes the absolute values of the weights. This can lead to sparsity, driving some weights to exactly zero, effectively performing feature selection by "turning off" less important features or nodes in the network.
                                   Useful when there are many features and some might be irrelevant, as it can effectively perform feature selection.
        L2 Regularization (Ridge): Penalizes the squared values of the weights. This shrinks the weights but generally doesn't force them to zero. This helps to prevent individual weights from becoming excessively large and dominating the model.
                                   Generally preferred in deep learning for its ability to smoothly reduce weight magnitudes and improve model generalization without completely removing features.
        """
        self._samples = samples
        self.model = Sequential([
                Input(shape=(self._samples,2)),
                Dense(16, activation='relu', name="L1", kernel_regularizer=l2(0.01)), # Decrease to fix high bias; Increase to fix high variance.
                Dense(32, activation='relu', name="L2", kernel_regularizer=l2(0.01)),
                # Just compute z. Puts both the activation function g(z) and cross entropy loss into the specification of the loss function below. This gives less roundoff error.
                # The output will consist of a vector with two elements that can be any value ranging from negative infinity to infinity, which will represent (x̃₁, x̃₂).
                Dense(2, name="L3")]) # Linear activation ("pass-through") if not specified
        print(f"Generator L3 output shape: {self.model.output_shape}")
        assert self.model.output_shape == (None, self._samples, 2)
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
        return self.model(input, training = training)

    def loss(self, real, fake):
        """
        This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
        """
        return self._cross_entropy(tf.ones_like(fake), fake)
    
    def UpdateParameters(self, tape, loss):
        # Use the gradient tape to automatically retrieve the gradients of the loss with respect to the trainable variables, dJ/dw.
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Run one step of gradient descent by updating the value of the variable to minimize the loss
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

class SineWaveGAN():
    """
    https://realpython.com/generative-adversarial-networks/
    What Are Generative Adversarial Networks?
    Generative adversarial networks are machine learning systems that can learn to mimic a given distribution of data. They were first proposed in a 2014 NeurIPS paper by deep learning expert Ian Goodfellow and his colleagues.

    GANs consist of two neural networks, one trained to generate data and the other trained to distinguish fake data from real data (hence the “adversarial” nature of the model). Although the idea of a structure to generate data isn’t new, when it comes to image and video generation, GANs have provided impressive results such as:

    Style transfer using CycleGAN, which can perform a number of convincing style transformations on images
    Generation of human faces with StyleGAN, as demonstrated on the website This Person Does Not Exist
    Structures that generate data, including GANs, are considered generative models in contrast to the more widely studied discriminative 
    """
    _num_sine_waves: int = None
    _samples: int = None
    _generator: Generator = None
    _discriminator: Discriminator = None
    def __init__(self, num_sine_waves:int, batch_size:int, samples:int, epochs:int, checkpoint_path: str):
        self._num_sine_waves = num_sine_waves
        self._batch_size = batch_size
        self._samples = samples
        self._epochs = epochs
        self._noise_dim = 100
        self._checkpoint_path = checkpoint_path
        self._discriminator = Discriminator(self._samples)
        self._generator = Generator(self._samples)
        self._checkpoint = tf.train.Checkpoint(generator_optimizer = self._generator.optimizer,
                                        discriminator_optimizer = self._discriminator.optimizer,
                                        generator = self._generator.model,
                                        discriminator = self._discriminator.model)
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    # @tf.function decorator to increase performance. Without this decorator our training will take twice as long. If you would like to know more about how to increase performance with @tf.function take a look at the TensorFlow documentation.
    @tf.function
    def _TrainStep(self, data_batch):
        """
        The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
        The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
        """
        noise = numpy.empty((data_batch.shape[0], self._samples, 2))
        noise[:,:,0] = rng.uniform(low=0, high=(2 * math.pi), size=self._samples)
        noise[:,:,1] = rng.uniform(low=0.1, high=0.9) * rng.standard_normal(noise[0,:,0].shape[0])
        # TensorFlow has the marvelous capability of calculating the derivatives for you. This is shown below. Within the tf.GradientTape() section, operations on Tensorflow Variables are tracked. When tape.gradient() is later called, it will return the gradient of the loss relative to the tracked variables. The gradients can then be applied to the parameters using an optimizer.
        # Tensorflow GradientTape records the steps used to compute cost J to enable auto differentiation.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self._generator.forward(noise, training=True)
            real_output = self._discriminator.forward(data_batch, training=True)
            fake_output = self._discriminator.forward(generated_images, training=True)

            gen_loss = self._generator.loss(real_output, fake_output)
            disc_loss = self._discriminator.loss(real_output, fake_output)

        self._generator.UpdateParameters(gen_tape, gen_loss)
        # Use the GradientTape to calculate the gradients of the cost with respect to the parameter w: dJ/dw.
        self._discriminator.UpdateParameters(disc_tape, disc_loss)
        return gen_loss, disc_loss

    def Train(self):
        # Reuse this seed overtime so that it's easier to visualize progress in the animated GIF
        seed = numpy.empty((1, self._samples, 2))
        seed[0,:,0] = rng.uniform(low=0, high=(2 * math.pi), size=self._samples)
        seed[:,:,1] = rng.uniform(low=0.1, high=0.9) * rng.standard_normal(seed[0,:,0].shape[0])
        """
        The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
        The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
        """
        gen_losses = []
        disc_losses = []
        for epoch in tqdm(range(self._epochs)):
            start = time.time()
            ave_gen_loss = 0
            ave_disc_loss = 0
            count = 0
            for data_batch in self._batch_dataset:
                gen_loss, disc_loss = self._TrainStep(data_batch)
                count += 1
                ave_gen_loss += gen_loss
                ave_disc_loss += disc_loss

            # Produce images for the GIF as you go
            self._save_images(self._generator.forward(seed, training=False), f"Generated Image at Epoch {epoch}", f'sinewave_gan_epoch_{epoch+1:04d}.png')

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self._checkpoint.save(file_prefix = self._checkpoint_path)

            ave_gen_loss /= count
            ave_disc_loss /= count
            gen_losses.append(ave_gen_loss)
            disc_losses.append(ave_disc_loss)
            print(f"Epoch {epoch + 1} : {time.time()-start}s Generator Loss: {ave_gen_loss} Discriminator Loss: {ave_disc_loss}")

        # Generate after the final epoch
        PlotGANLossHistory("Sine Wave GAN", gen_losses, disc_losses)
        self._save_images(self._generator.forward(seed, training=False), f"Generated Image at Epoch {self._epochs}", f'sinewave_gan_epoch_{self._epochs:04d}.png')

    def _save_images(self, data, title:str, filename: str):
        # Notice `training` is set to False. This is so all layers forward in inference mode (batchnorm).
        #print(f"save_images data.shape: {data.shape}, ndim: {data.ndim}") # data.shape: (1, 1024, 2), ndim: 3
        plt.plot(data[0, :, 0], data[0, :, 1], ".", color="blue")
        plt.suptitle(title, fontsize=22, fontweight="bold")
        plt.savefig(f"output/SineWaveGAN/{filename}")
        plt.close()

    def _ShowSineWaves(self, data):
        fig, axes = plt.subplots(3,3, constrained_layout=True, figsize=(10,10))
        fig.tight_layout(pad=5, rect=[0, 0.03, 1, 0.95]) #[left, bottom, right, top]
        for i, ax in enumerate(axes.flat):
            index = rng.choice(len(data))
            ax.plot(data[index, :, 0], data[index, :, 1], ".", color="blue")
            ax.set_axis_off()
        fig.suptitle("Noisy Sine Waves", fontsize=22, fontweight="bold")
        plt.show()

    def PrepareTrainingData(self):
        """
        The training data is composed of pairs (x₁, x₂) so that x₂ consists of the value of the sine of x₁ for x₁ in the interval from 0 to 2π.
        """
        dataset = numpy.empty((self._num_sine_waves, self._samples, 2))
        for i in range(self._num_sine_waves):
            dataset[i, :, 0] = rng.uniform(low=0, high=(2 * math.pi), size = self._samples)
            #dataset[i, :, 1] = numpy.sin(dataset[i, :, 0]) + (rng.uniform(low=0.1, high=0.9) * rng.standard_normal(dataset[i, :, 0].shape[0])) Fails to converge when large multiplier values
            dataset[i, :, 1] = numpy.sin(dataset[i, :, 0]) + (0.1 * rng.standard_normal(dataset[i, :, 0].shape[0]))
        self._ShowSineWaves(dataset)
        assert dataset.shape == (self._num_sine_waves, self._samples, 2)
        print(f"dataset type: {type(dataset)}, shape: {dataset.shape}") # <class 'numpy.ndarray'>, shape: (10000, 1024, 2)
        self._batch_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(dataset)).shuffle(self._num_sine_waves, reshuffle_each_iteration=True).batch(self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE) # train_dataset: TensorSpec(shape=(None, 2), dtype=tf.float64, name=None)

if __name__ == "__main__":
    NUM_SINE_WAVES = 25000
    SAMPLES = 1024
    BATCH_SIZE = 100
    EPOCHS = 150
    InitializeGPU()
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "sinewave_gan")
    Path("output/SineWaveGAN").mkdir(parents=True, exist_ok=True)
    Path("output/SineWaveGAN").is_dir()
    sinewaveGAN = SineWaveGAN(NUM_SINE_WAVES, BATCH_SIZE, SAMPLES, EPOCHS, checkpoint_prefix)
    sinewaveGAN.PrepareTrainingData()
    sinewaveGAN.Train()
    ShowImage(f'output/SineWaveGAN/sinewave_gan_epoch_{EPOCHS:04d}.png')
    CreateGIF("output/SineWaveGAN/sinewave_gan.gif", 'output/SineWaveGAN/sinewave_gan_epoch_*.png')