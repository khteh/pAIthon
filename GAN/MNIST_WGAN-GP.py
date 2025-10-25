import matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf
from pathlib import Path
from GAN.MNISTGAN import Generator as MNIST_Generator, Discriminator, MNISTGAN
from utils.Image import ShowImage, CreateGIF
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotGANLossHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
# https://www.tensorflow.org/tutorials/generative/dcgan
# https://jonathan-hui.medium.com/gan-how-to-measure-gan-performance-64b988c47732
class Critic(Discriminator):
    _lambda: float = None # Weight of the gradient penalty
    def __init__(self, _lambda: float):
        self._lambda = _lambda
        super().__init__()

    def loss(self, real, fake, mixed_images, mixed_score, tape):
        """
        https://ai.stackexchange.com/questions/25411/wasserstein-gan-implemention-of-critic-loss-correct
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 

        This method quantifies how well the critic is able to distinguish real images from fakes. It compares the critic's predictions on real images to an array of 1s, and the critic's predictions on fake (generated) images to an array of 0s.
        """
        # E(c(x)) - E(c(g(x))) + lambda * regularization
        # Adam will minimize the loss. So, by minimizing -(torch.mean(crit_real_pred) - torch.mean(crit_fake_pred)),
        # it effectively maximizes the distance between the real and fake distributions.
        gradients = tape.gradient(mixed_score, mixed_images)
        penalty = self._gradient_penalty(gradients)
        loss = -(tf.reduce_mean(real) - tf.reduce_mean(fake)) + penalty * self._lambda
        #print(f"real: {tf.reduce_mean(real)}, fake: {tf.reduce_mean(fake)}, regularization: {penalty * self._lambda}, loss: {loss}")
        return loss

    def _gradient_penalty(self, gradient):
        '''
        Return the gradient penalty, given a gradient.
        Given a batch of image gradients, you calculate the magnitude of each image's gradient
        and penalize the mean quadratic distance of each magnitude to 1.
        Parameters:
            gradient: the gradient of the critic's scores, with respect to the mixed image
        Returns:
            penalty: the gradient penalty
        '''
        # Calculate the magnitude of every row
        gradient_norm = tf.math.l2_normalize(gradient, axis=1)
        #print(f"gradient_norm: {gradient_norm.shape}")
        # Penalize the mean squared distance of the gradient norms from 1
        penalty = tf.math.reduce_sum(tf.math.square(tf.math.subtract(gradient_norm, 1))) / gradient_norm.shape[0]
        #print(f"penalty: {penalty.shape} {penalty}")
        return penalty

class WGANGenerator(MNIST_Generator):
    def loss(self, real, fake):
        """
        The generator's loss quantifies how well it was able to trick the critic. Intuitively, if the generator is performing well, the critic will classify the fake images as real (or 1). Here, compare the critics decisions on the generated images to an array of 1s.
        The generator tries to maximize the scores (or equivalently minimize the negative.
        """
        return - tf.math.reduce_mean(fake)

class MNIST_WGAN_GP(MNISTGAN):
    """
    WGAN-GP isn't necessarily meant to improve overall performance of a GAN, but just **increases stability** and avoids mode collapse. 
    In general, a WGAN will be able to train in a much more stable way than the vanilla DCGAN from last assignment, though it will generally forward a bit slower. 
    You should also be able to train your model for more epochs without it collapsing.
    """
    _critic: Critic = None
    _critic_repeats: int = None # number of times to update the critic per generator update
    def __init__(self, name:str, _lambda: float, _critic_repeats: int, buffer_size:int, batch_size:int, epochs:int, checkpoint_path: str):
        super().__init__(name, buffer_size, batch_size, epochs, checkpoint_path)
        self._critic_repeats = _critic_repeats
        self._generator = WGANGenerator()
        self._critic = Critic(_lambda)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    # @tf.function decorator to increase performance. Without this decorator our training will take twice as long. If you would like to know more about how to increase performance with @tf.function take a look at the TensorFlow documentation.
    @tf.function
    def _TrainStep(self, images):
        """
        The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The critic is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
        The loss is calculated for each of these models, and the gradients are used to update the generator and critic.
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
        """
        #print(f"\n=== MNIST_WGAN_GP._TrainStep ===")
        mean_critic_loss = 0
        for _ in range(self._critic_repeats):
            noise = rng.random([images.shape[0], self._noise_dim])
            #print(f"noise: {noise.shape}")
            # TensorFlow has the marvelous capability of calculating the derivatives for you. This is shown below. Within the tf.GradientTape() section, operations on Tensorflow Variables are tracked. When tape.gradient() is later called, it will return the gradient of the loss relative to the tracked variables. The gradients can then be applied to the parameters using an optimizer.
            # Tensorflow GradientTape records the steps used to compute cost J to enable auto differentiation.
            # A non-persistent GradientTape can only be used to compute one set of gradients (or jacobians)
            with tf.GradientTape(persistent=True) as tape:
                generated_images = self._generator.forward(noise, training=True)
                # Mix the images together
                epsilon = tf.random.uniform(shape=[images.shape[0], 1, 1, 1]) #torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                #print(f"epsilon: {epsilon.shape}, images: {images.shape}, generated: {generated_images.shape}")
                mixed_images = tf.math.multiply(images, epsilon) + tf.math.multiply(generated_images,  (1 - epsilon))
                # Calculate the critic's scores on the mixed images
                real_output = self._critic.forward(images, training=True)
                fake_output = self._critic.forward(generated_images, training=True)
                mixed_scores = self._critic.forward(mixed_images, training=True)
                # Take the gradient of the scores with respect to the images
                # Calculate the gradients of z with respect to x and y
                loss = self._critic.loss(real_output, fake_output, mixed_images, mixed_scores, tape) # This must be done inside the scope / context of the GradientTape to record the gradient of the loss variable.
            # Keep track of the average critic loss in this batch
            mean_critic_loss += loss
            self._critic.UpdateParameters(tape, loss)

        with tf.GradientTape() as tape:
            generated_images = self._generator.forward(noise, training=True)
            real_output = self._critic.forward(images, training=True)
            fake_output = self._critic.forward(generated_images, training=True)
            gen_loss = self._generator.loss(real_output, fake_output)
        self._generator.UpdateParameters(tape, gen_loss)
        # Use the GradientTape to calculate the gradients of the cost with respect to the parameter w: dJ/dw.
        return gen_loss, mean_critic_loss / self._critic_repeats

if __name__ == "__main__":
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 100
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "mnist_wgan_gp")
    Path("output/MNIST_WGAN_GP").mkdir(parents=True, exist_ok=True)
    Path("output/MNIST_WGAN_GP").is_dir()
    InitializeGPU()
    mnistGAN = MNIST_WGAN_GP("mnist_wgan", 10, 5, BUFFER_SIZE, BATCH_SIZE, EPOCHS, checkpoint_prefix)
    mnistGAN.Train(16, 4, 4)
    ShowImage(f'output/MNIST_WGAN_GP/mnist_wgan_epoch_{EPOCHS:04d}.png')
    CreateGIF("output/MNIST_WGAN_GP/mnist_wgan.gif", 'output/MNIST_WGAN_GP/mnist_wgan_epoch_*.png')