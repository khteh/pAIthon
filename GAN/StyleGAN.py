import matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Flatten, Dense, BatchNormalization, Reshape, Conv2DTranspose
from scipy.stats import truncnorm
from tensorflow.keras import layers, losses, optimizers, regularizers
from utils.Image import ShowImage, CreateGIF
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotGANLossHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

class MappingLayers(tf.Module):
    '''
    Mapping Layers Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        hidden_dim: the inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
    _z_dim: int = None
    _w_dim: int = None
    _hidden_dim: int = None
    _noise_mapping = None
    def __init__(self, z_dim:int, hidden_dim:int, w_dim:int):
        # Input shape (n_samples, z_dim) and outputs (n_samples, w_dim)
        # with a hidden layer with hidden_dim neurons
        super().__init__()
        self._z_dim = z_dim
        self._w_dim = w_dim
        self._hidden_dim = hidden_dim
        self._noise_mapping = self.model = Sequential([
            Input(shape=(self._z_dim,)),
            Dense(self._hidden_dim, activation="relu"),
            Dense(self._hidden_dim, activation="relu"),
            Dense(self._w_dim),
        ])
    def __call__(self, noise):
        return self._noise_mapping(noise)

class InjectNoise(tf.Module):
    '''
    Inject Noise Class
    Values:
        channels: the number of channels the image has, a scalar
    '''
    _weights: tf.Variable = None
    _channels:int = None
    def __init__(self, channels:int):
        super().__init__()
        self._channels = channels
        self._weights = tf.Variable( # You use nn.Parameter so that these weights can be optimized
            # Initiate the weights for the channels from a random normal distribution
            tf.random.normal(shape=(1, self._channels, 1, 1)),
            name = "NoiseWeights"
        )
    def __call__(self, image):
        '''
        Inject Noise Class
        Values:
            channels: the number of channels the image has, a scalar
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''
        # Set the appropriate shape for the noise!
        # You would first create a random (4 x 4) noise matrix with one channel.
        noise = tf.random.normal(shape=(image.shape[0], 1, image.shape[2], image.shape[3]))
        weight_noise = self._weights * noise
        #print(f"image: {image.shape}, weight: {self._weights.shape}, noise: {noise.shape}, weight_noise: {weight_noise.shape}")
        return image + self._weights * noise
    
    def GetNoiseWeights(self):
        return self._weights
    
    def SetNoiseWeights(self, weights):
        self._weights = tf.Variable(weights, name = "NoiseWeights")
        return self._weights

class AdaIN(tf.Module):
    '''
    AdaIN Class
    Values:
        channels: the number of channels the image has, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
    _channels:int = None
    _w_dim:int = None
    _instance_norm = None
    style_scale_transform: Dense = None
    style_shift_transform:Dense = None
    def __init__(self, channels: int, w_dim: int):
        super().__init__()
        self._channels = channels
        self._w_dim = w_dim
        # Normalize the input per-dimension
        self._instance_norm = BatchNormalization()

        # You want to map w to a set of style weights per channel.
        # Replace the Nones with the correct dimensions - keep in mind that 
        # both linear maps transform a w vector into style weights 
        # corresponding to the number of image channels.
        self.style_scale_transform = Dense(channels)
        self.style_shift_transform = Dense(channels)
    def __call__(self, image, w):
        '''
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w, 
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        normalized_image = self._instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        
        # Calculate the transformed image
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image        
    
class StyleGAN():
    _noise_mapping = None
    _z_dim: int = None
    _w_dim: int = None
    _hidden_dim: int = None
    _channels: int = None
    def __init__(self, z_dim:int, hidden_dim:int, w_dim:int, channels: int):
        self._z_dim = z_dim
        self._w_dim = w_dim
        self._hidden_dim = hidden_dim
        self._channels = channels
       
    def get_truncated_noise(self, n_samples, z_dim, truncation):
        '''
        Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
        and truncation value, creates a tensor of that shape filled with random
        numbers from the truncated normal distribution.
        Parameters:
            n_samples: the number of samples to generate, a scalar
            z_dim: the dimension of the noise vector, a scalar
            truncation: the truncation value, a non-negative scalar
        '''
        truncated_noise = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim))
        #print(f"truncated_noise: {truncated_noise.shape}")
        return truncated_noise

def AdaINTests():
    print(f"\n=== AdaINTests ===")
    w_channels = 50
    image_channels = 20
    image_size = 30
    n_test = 10
    adain = AdaIN(image_channels, w_channels)
    test_w = tf.random.normal(shape=(n_test, w_channels))
    assert adain.style_scale_transform(test_w).shape == adain.style_shift_transform(test_w).shape
    assert adain.style_scale_transform(test_w).shape[-1] == image_channels
    assert tuple(adain(tf.random.normal(shape=(n_test, image_channels, image_size, image_size)), test_w).shape) == (n_test, image_channels, image_size, image_size)

    w_channels = 3
    image_channels = 2
    image_size = 3
    n_test = 1
    adain = AdaIN(image_channels, w_channels)

    adain.style_scale_transform.set_weights(tf.ones_like(adain.style_scale_transform.weights) / 4)
    adain.style_scale_transform.use_bias(tf.zeros_like(adain.style_scale_transform.bias))
    adain.style_shift_transform.set_weights(tf.ones_like(adain.style_shift_transform.weights) / 5)
    adain.style_shift_transform.use_bias(tf.zeros_like(adain.style_shift_transform.bias))
    test_input = tf.ones(n_test, image_channels, image_size, image_size)
    test_input[:, :, 0] = 0
    test_w = tf.ones(n_test, w_channels)
    test_output = adain(test_input, test_w)
    assert(tf.math.abs(test_output[0, 0, 0, 0] - 3 / 5 + tf.math.sqrt(tf.tensor(9 / 8))) < 1e-4)
    assert(tf.math.abs(test_output[0, 0, 1, 0] - 3 / 5 - tf.math.sqrt(tf.tensor(9 / 32))) < 1e-4)
    print("\n\033[92mAll test passed!")

def InjectNoiseTests():
    print(f"\n=== InjectNoiseTests ===")
    inject_noise = InjectNoise(3000)

    test_noise_channels = 3000
    test_noise_samples = 20
    fake_images = tf.random.normal(shape=(test_noise_samples, test_noise_channels, 10, 10))
    weights = inject_noise.GetNoiseWeights()
    assert tf.math.abs(tf.math.reduce_std(weights) - 1) < 0.1
    assert tf.math.abs(tf.math.reduce_mean(weights)) < 0.1

    assert tuple(weights.shape) == (1, test_noise_channels, 1, 1)
    weights = inject_noise.SetNoiseWeights(tf.ones_like(weights))
    # Check that something changed
    assert tf.math.reduce_mean(tf.math.abs((inject_noise(fake_images) - fake_images))) > 0.1
    # Check that the change is per-channel
    for i in range(4):
        print(f"{i}: {tf.math.reduce_mean(tf.math.abs(tf.math.reduce_std(inject_noise(fake_images) - fake_images, axis=i)))}")
    assert tf.math.reduce_mean(tf.math.abs(tf.math.reduce_std(inject_noise(fake_images) - fake_images, axis=0))) > 1e-4
    #print(f"{tf.math.abs((inject_noise(fake_images) - fake_images).std(1)).mean()}")
    assert tf.math.reduce_mean(tf.math.abs(tf.math.reduce_std(inject_noise(fake_images) - fake_images, axis=1))) < 1e-4
    assert tf.math.reduce_mean(tf.math.abs(tf.math.reduce_std(inject_noise(fake_images) - fake_images, axis=2))) > 1e-4
    assert tf.math.reduce_mean(tf.math.abs(tf.math.reduce_std(inject_noise(fake_images) - fake_images, axis=3))) > 1e-4
    # Check that the per-channel change is roughly normal
    per_channel_change = tf.math.reduce_std(tf.math.reduce_mean(inject_noise(fake_images) - fake_images, axis=1))
    print(f"per_channel_change: {per_channel_change}")
    assert per_channel_change > 0.9 and per_channel_change < 1.1
    # Make sure that the weights are being used at all
    weights = inject_noise.SetNoiseWeights(tf.zeros_like(weights))
    assert tf.math.abs((tf.math.reduce_mean(inject_noise(fake_images) - fake_images))) < 1e-4
    assert len(weights.shape) == 4
    print("\n\033[92mAll test passed!")

def NoiseMappingLayersTests():
    print(f"\n=== NoiseMappingLayersTests ===")
    mapping = MappingLayers(10,20,30)
    assert tuple(mapping(tf.random.normal(shape=(2, 10))).shape) == (2, 30)
    #assert len(mapping.mapping) > 4 # Check number of sequential layers
    outputs = mapping(tf.random.normal(shape=(1000, 10)))
    print(f"std: {tf.math.reduce_std(outputs)}, min: {tf.math.reduce_min(outputs)}, max: {tf.math.reduce_max(outputs)}")
    #assert tf.math.reduce_std(outputs) > 0.05 and tf.math.reduce_std(outputs) < 0.3
    #assert tf.math.reduce_min(outputs) > -2 and tf.math.reduce_min(outputs) < 0
    #assert tf.math.reduce_max(outputs) < 2 and tf.math.reduce_max(outputs) > 0

def truncated_noise_tests():
    print(f"\n=== truncated_noise_tests ===")
    style_gan = StyleGAN(10,20,30, 3000)
    # Test the truncation sample
    assert tuple(style_gan.get_truncated_noise(n_samples=10, z_dim=5, truncation=0.7).shape) == (10, 5)
    simple_noise = style_gan.get_truncated_noise(n_samples=1000, z_dim=10, truncation=0.2)
    assert simple_noise.max() > 0.199 and simple_noise.max() < 2
    assert simple_noise.min() < -0.199 and simple_noise.min() > -0.2
    assert simple_noise.std() > 0.113 and simple_noise.std() < 0.117
    print("\n\033[92mAll test passed!")

if __name__ == "__main__":
    truncated_noise_tests()
    NoiseMappingLayersTests()
    InjectNoiseTests()
    AdaINTests()