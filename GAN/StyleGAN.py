import matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf
from pathlib import Path
from tensorflow.image import ResizeMethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Flatten, Dense, GroupNormalization, Reshape, Conv2DTranspose, UpSampling2D
from scipy.stats import truncnorm
from tensorflow.keras import layers, losses, optimizers, regularizers
from utils.Image import ShowImage, CreateGIF, make_image_grid
from utils.GPU import InitializeGPU, SetMemoryLimit
from utils.TrainingMetricsPlot import PlotGANLossHistory
from utils.TermColour import bcolors
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
# https://jonathan-hui.medium.com/gan-how-to-measure-gan-performance-64b988c47732
# https://jonathan-hui.medium.com/gan-stylegan-stylegan2-479bdf256299
def get_truncated_noise(n_samples, z_dim, truncation):
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
        # self._instance_norm = BatchNormalization(axis=1) # axis 'channels'
        # Option 1: Using GroupNormalization to mimic InstanceNormalization
        # For InstanceNormalization, groups are typically set to the number of channels.
        self._instance_norm = GroupNormalization(
            groups=self._channels, # Set groups to the number of channels for Instance Normalization effect
            axis=1, # channels-first format
            epsilon=1e-5,
            center=True,
            scale=True,
        )
        # You want to map w to a set of style weights per channel.
        # Replace the Nones with the correct dimensions - keep in mind that 
        # both linear maps transform a w vector into style weights 
        # corresponding to the number of image channels.
        self.style_scale_transform = Dense(channels) # dimensionality of the output space.
        self.style_shift_transform = Dense(channels)
    def __call__(self, image, w):
        '''
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w, 
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        print(f"=== AdaIN.__call__ ===")
        normalized_image = self._instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        #print(f"image: {tf.math.reduce_mean(image)}, normalized_image: {tf.math.reduce_mean(normalized_image)}, style_scale: {tf.math.reduce_mean(style_scale)}, style_shift: {tf.math.reduce_mean(style_shift)}")
        # Calculate the transformed image
        return style_scale * normalized_image + style_shift
    
class MicroStyleGANGeneratorBlock(tf.Module):
    '''
    Micro StyleGAN Generator Block Class
    Values:
        in_chan: the number of channels in the input, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        kernel_size: the size of the convolving kernel
        starting_size: the size of the starting image
    '''
    _in_chan: int = None
    _out_chan: int = None
    _w_dim: int = None
    _kernel_size: int = None
    _factor: int = None
    _use_upsample: bool = None
    upsample: UpSampling2D = None
    conv: Conv2D = None
    inject_noise: InjectNoise = None
    adain : AdaIN = None
    activation: LeakyReLU = None
    def __init__(self, in_chan:int, out_chan:int, w_dim:int, kernel_size:int, factor:int, use_upsample:bool=True):
        super().__init__()
        self._in_chan = in_chan
        self._out_chan = out_chan
        self._w_dim = w_dim
        self._kernel_size = kernel_size
        self._factor = factor
        self._use_upsample = use_upsample
        # Replace the Nones in order to:
        # 1. Upsample to the starting_size, bilinearly (https://pytorch.org/docs/master/generated/torch.nn.Upsample.html)
        # 2. Create a kernel_size convolution which takes in 
        #    an image with in_chan and outputs one with out_chan (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
        # 3. Create an object to inject noise
        # 4. Create an AdaIN object
        # 5. Create a LeakyReLU activation with slope 0.2
        if self._use_upsample:
            self.upsample = UpSampling2D(factor, interpolation='bilinear', data_format="channels_first") # size: The upsampling factors for rows and columns.
        self.conv = Conv2D(out_chan, kernel_size, padding="same", data_format="channels_first")
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan, w_dim)
        self.activation = LeakyReLU(0.2)

    def __call__(self, x, w):
        '''
        Function for completing a forward pass of MicroStyleGANGeneratorBlock: Given an x and w, 
        computes a StyleGAN generator block.
        Parameters:
            x: the input into the generator, feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        if self._use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.adain(x, w)
        x = self.activation(x)
        return x

class MicroStyleGANGenerator(tf.Module):
    '''
    Micro StyleGAN Generator Class.
    StyleGAN starts with a constant 4 x 4 (x 512 channel) tensor which is put through an iteration of the generator without upsampling. The output is some noise that can then be transformed into a blurry 4 x 4 image. This is where the progressive growing process begins. The 4 x 4 noise can be further passed through a generator block with upsampling to produce an 8 x 8 output. However, this will be done gradually.
    You will simulate progressive growing from an 8 x 8 image to a 16 x 16 image. Instead of simply passing it to the generator block with upsampling, StyleGAN gradually trains the generator to the new size by mixing in an image that was only upsampled. By mixing an upsampled 8 x 8 image (which is 16 x 16) with increasingly more of the 16 x 16 generator output, the generator is more stable as it progressively trains. As such, you will do two separate operations with the 8 x 8 noise:
    1.   Pass it into the next generator block to create an output noise, that you will then transform to an image.
    2.   Transform it into an image and then upsample it to be 16 x 16.
    You will now have two images that are both double the resolution of the 8 x 8 noise. Then, using an alpha ($\alpha$) term, you combine the higher resolution images obtained from (1) and (2). You would then pass this into the discriminator and use the feedback to update the weights of your generator. The key here is that the $\alpha$ term is gradually increased until eventually, only the image from (1), the generator, is used. That is your final image or you could continue this process to make a 32 x 32 image or 64 x 64, 128 x 128, etc. 
    This micro model you will implement will visualize what the model outputs at a particular stage of training, for a specific value of $\alpha$. However to reiterate, in practice, StyleGAN will slowly phase out the upsampled image by increasing the $\alpha$ parameter over many training steps, doing this process repeatedly with larger and larger alpha values until it is 1—at this point, the combined image is solely comprised of the image from the generator block. This method of gradually training the generator increases the stability and fidelity of the model.

    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        in_chan: the dimension of the constant input, usually w_dim, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    '''
    _z_dim:int = None
    _map_hidden_dim = None
    _w_dim: int = None
    _in_chan: int = None
    _out_chan: int = None
    _kernel_size: int = None
    _hidden_chan: int = None
    alpha:float = None
    def __init__(self, 
                 z_dim, 
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan, 
                 kernel_size, 
                 hidden_chan,
                 alpha:float):
        super().__init__()
        self._z_dim = z_dim
        self._map_hidden_dim = map_hidden_dim
        self._w_dim = w_dim
        self._in_chan = in_chan
        self._out_chan = out_chan
        self._kernel_size = kernel_size
        self._hidden_chan = hidden_chan
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        # Typically this constant is initiated to all ones, but you will initiate to a
        # Gaussian to better visualize the network's effect
        self.starting_constant = tf.Variable(tf.random.normal((1, in_chan, 4, 4)))
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)
        # You need to have a way of mapping from the output noise to an image, 
        # so you learn a 1x1 convolution to transform the e.g. 512 channels into 3 channels
        # (Note that this is simplified, with clipping used in the real StyleGAN)
        self.block1_to_image = Conv2D(out_chan, kernel_size=1)
        self.block2_to_image = Conv2D(out_chan, kernel_size=1)
        self.alpha = alpha

    def upsample_to_match_size(self, smaller_image, bigger_image):
        '''
        Function for upsampling an image to the size of another: Given a two images (smaller and bigger), 
        upsamples the first to have the same dimensions as the second.
        Parameters:
            smaller_image: the smaller image to upsample
            bigger_image: the bigger image whose dimensions will be upsampled to
        '''
        return tf.image.resize(smaller_image, size=bigger_image.shape[-2:], method=ResizeMethod.BILINEAR)
        
    def __call__(self, noise, return_intermediate=False):
        '''
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise, 
        computes a StyleGAN iteration.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
            return_intermediate: a boolean, true to return the images as well (for testing) and false otherwise
        '''
        x = self.starting_constant
        w = self.map(noise)
        x = self.block0(x, w)
        x_small = self.block1(x, w) # First generator run output
        x_small_image = self.block1_to_image(x_small)
        x_big = self.block2(x_small, w) # Second generator run output 
        x_big_image = self.block2_to_image(x_big)
        x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image) # Upsample first generator run output to be same size as second generator run output 
        # Interpolate between the upsampled image and the image from the generator using alpha
        interpolation = self._lerp(x_small_upsample, x_big_image, self.alpha)
        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
        return interpolation

    def _lerp(self, start, end, weight):
        """
        Performs linear interpolation between two tensors, start and end,
        based on a scalar or tensor weight.
        https://docs.pytorch.org/docs/stable/generated/torch.lerp.html
        
        Args:
            start: The tensor with the starting points.
            end: The tensor with the ending points.
            weight: A float or tensor representing the weight for the interpolation.

        Returns:
            A tensor representing the interpolated values.
        """
        return start + weight * (end - start)    
class StyleGAN():
    _samples = None
    _truncation = None
    _z_dim:int = None
    _map_hidden_dim = None
    _w_dim: int = None
    _in_chan: int = None
    _out_chan: int = None
    _kernel_size: int = None
    _hidden_chan: int = None
    _alpha:float = None
    _stylegan_generator: MicroStyleGANGenerator = None
    _noise = None
    def __init__(self, samples:int, truncation, z_dim, 
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan, 
                 kernel_size, 
                 hidden_chan,
                 alpha:float):
        self._samples = samples
        self._truncation = truncation
        self._z_dim = z_dim
        self._map_hidden_dim = map_hidden_dim
        self._w_dim = w_dim
        self._in_chan = in_chan
        self._out_chan = out_chan
        self._kernel_size = kernel_size
        self._hidden_chan = hidden_chan
        self._alpha = alpha
        self._stylegan_generator = MicroStyleGANGenerator(
            z_dim=z_dim, 
            map_hidden_dim=map_hidden_dim,
            w_dim=w_dim,
            in_chan=in_chan,
            out_chan=out_chan, 
            kernel_size=kernel_size, 
            hidden_chan=hidden_chan,
            alpha=alpha
        )
        self._noise = get_truncated_noise(self._samples, self._z_dim, self._truncation) * 10

    def Run(self):
        print(f"\n=== {self.Run.__name__} ===")
        #self._stylegan_generator.evaluate() https://discuss.ai.google.dev/t/tensorflow-keras-equivalent-of-pytorch-nn-module-eval/108262
        images = []
        for alpha in numpy.linspace(0, 1, num=5):
            self._stylegan_generator.alpha = alpha
            viz_result, _, _ =  self._stylegan_generator(
                self._noise, 
                return_intermediate=True)
            images += [tensor for tensor in viz_result]
        self._show_tensor_images(tf.stack(images), nrow=self._samples, num_images=len(images))
        self._stylegan_generator = self._stylegan_generator.train()        

    def _show_tensor_images(self, image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
        '''
        Function for visualizing images: Given a tensor of images, number of images,
        size per image, and images per row, plots and prints the images in an uniform grid.
        '''
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
        image_grid = make_image_grid(image_unflat[:num_images], nrow=nrow, padding=0)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.axis('off')
        plt.show()

def StyleGANTests():
    print(f"\n=== {StyleGANTests.__name__} ===")
    z_dim = 128
    out_chan = 3
    truncation = 0.7
    alpha=0.2
    styleGAN = StyleGAN(10, truncation = truncation, z_dim=z_dim, 
        map_hidden_dim=1024,
        w_dim=496,
        in_chan=512,
        out_chan=out_chan, 
        kernel_size=3, 
        hidden_chan=256,
        alpha=alpha
    )
    styleGAN.Run()

def MicroStyleGANGeneratorTests():
    print(f"\n=== {MicroStyleGANGeneratorTests.__name__} ===")
    z_dim = 128
    out_chan = 3
    truncation = 0.7
    alpha=0.2
    stylegan_generator = MicroStyleGANGenerator(
        z_dim=z_dim, 
        map_hidden_dim=1024,
        w_dim=496,
        in_chan=512,
        out_chan=out_chan, 
        kernel_size=3, 
        hidden_chan=256,
        alpha=alpha
    )
    test_samples = 10
    test_result = stylegan_generator(get_truncated_noise(test_samples, z_dim, truncation))

    # Check if the block works
    assert tuple(test_result.shape) == (test_samples, out_chan, 16, 16)

    # Check that the interpolation is correct
    stylegan_generator.alpha = 1.
    test_result, _, test_big =  stylegan_generator(
        get_truncated_noise(test_samples, z_dim, truncation), 
        return_intermediate=True)
    assert tf.abs(test_result - test_big).mean() < 0.001
    stylegan_generator.alpha = 0.
    test_result, test_small, _ =  stylegan_generator(
        get_truncated_noise(test_samples, z_dim, truncation), 
        return_intermediate=True)
    assert tf.abs(test_result - test_small).mean() < 0.001
    print(f"\n{bcolors.OKGREEN}All test passed!{bcolors.DEFAULT}")

def MicroStyleGANGeneratorBlockTests():
    print(f"\n=== {MicroStyleGANGeneratorBlockTests.__name__} ===")
    stylegan_generator_block = MicroStyleGANGeneratorBlock(in_chan=128, out_chan=64, w_dim=256, kernel_size=3, factor=2)
    test_x = numpy.ones((1, 128, 4, 4))
    test_x[:, :, 1:3, 1:3] = 0
    test_w = numpy.ones((1, 256))
    test_x = stylegan_generator_block.upsample(test_x)
    print(f"test_x.shape: {test_x.shape}")
    assert tuple(test_x.shape) == (1, 128, 8, 8)
    assert numpy.abs(tf.math.reduce_mean(test_x) - 0.75) < 1e-4
    test_x = stylegan_generator_block.conv(test_x)
    assert tuple(test_x.shape) == (1, 64, 8, 8)
    test_x = stylegan_generator_block.inject_noise(test_x)
    test_x = stylegan_generator_block.activation(test_x)
    assert tf.math.reduce_min(test_x) < 0
    assert -tf.math.reduce_min(test_x) / tf.math.reduce_max(test_x) < 0.4
    test_x = stylegan_generator_block.adain(test_x, test_w) 
    foo = stylegan_generator_block(tf.ones((10, 128, 4, 4)), tf.ones((10, 256)))
    print(f"\n{bcolors.OKGREEN}All test passed!{bcolors.DEFAULT}")

def AdaINTests():
    # https://github.com/tensorflow/tensorflow/issues/102870
    print(f"\n=== {AdaINTests.__name__} ===")
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
    test_input = numpy.ones((n_test, image_channels, image_size, image_size))
    test_input[:, :, 0] = 0
    test_w = tf.ones((n_test, w_channels))

    adain = AdaIN(image_channels, w_channels)
    adain.style_scale_transform(test_w)
    adain.style_shift_transform(test_w)
    style_scale_weights = adain.style_scale_transform.get_weights()
    style_shift_weights = adain.style_shift_transform.get_weights()
    style_scale_weights_kernel = numpy.ones_like(style_scale_weights[0]) / 4
    style_shift_weights_kernel = numpy.ones_like(style_shift_weights[0]) / 5
    style_scale_bias = numpy.zeros_like(style_scale_weights[1])
    style_shift_bias = numpy.zeros_like(style_shift_weights[1])
    adain.style_scale_transform.set_weights([style_scale_weights_kernel, style_scale_bias])
    adain.style_shift_transform.set_weights([style_shift_weights_kernel, style_shift_bias])
    # The bias (and kernel) attributes of a Dense layer are only created when the layer is built, which happens during the first call to the layer with input data or when the model containing the layer is compiled and trained.
    # If you try to access layer.bias before this, it won't exist.
    style_scale_weights = adain.style_scale_transform.get_weights()
    style_shift_weights = adain.style_shift_transform.get_weights()
    assert (style_scale_weights[0] == 1 / 4).all()
    assert (style_shift_weights[0] == 1 / 5).all()
    assert (style_scale_weights[1] == 0).all()
    assert (style_shift_weights[1] == 0).all()

    test_output = adain(test_input, test_w)
    print(f"test_input: {test_input.shape} {test_input}")
    print(f"test_output: {test_output.shape} {test_output}")
    # image: -0.0047212447971105576, normalized_image: -1.4834933281804297e-10, style_scale: -0.07495932281017303, style_shift: 0.034594107419252396
    # image: 0.6666666666666666, normalized_image: 0.0, style_scale: 0.75, style_shift: 0.6000000238418579
    assert(tf.math.abs(test_output[0, 0, 0, 0] - 3 / 5 + tf.math.sqrt(9 / 8)) < 1e-4)
    assert(tf.math.abs(test_output[0, 0, 1, 0] - 3 / 5 - tf.math.sqrt(9 / 32)) < 1e-4)
    print(f"\n{bcolors.OKGREEN}All test passed!{bcolors.DEFAULT}")

def InjectNoiseTests():
    print(f"\n=== {InjectNoiseTests.__name__} ===")
    # First, check the weights * stochastic noise broadcasting.
    weights = numpy.ones((1,3,1,1)) # 3 channels
    image = numpy.ones((5,3,10,10)) # 5 samples, 3 channels, w = h = 10
    noise = numpy.zeros(shape=(image.shape[0], 1, image.shape[2], image.shape[3]))
    weighted_noise = weights * noise
    noisy_image = image + weighted_noise
    print(f"noise: {noise.shape}")
    print(f"weihted_noise: {weighted_noise.shape}")
    print(f"noisy_image: {noisy_image.shape}")
    assert image.shape == weighted_noise.shape
    assert 0.0 == numpy.mean(weighted_noise)
    assert 1.0 == numpy.mean(noisy_image)

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
    print(f"\n{bcolors.OKGREEN}All test passed!{bcolors.DEFAULT}")

def NoiseMappingLayersTests():
    print(f"\n=== {NoiseMappingLayersTests.__name__} ===")
    mapping = MappingLayers(10,20,30)
    assert tuple(mapping(tf.random.normal(shape=(2, 10))).shape) == (2, 30)
    #assert len(mapping.mapping) > 4 # Check number of sequential layers
    outputs = mapping(tf.random.normal(shape=(1000, 10)))
    print(f"std: {tf.math.reduce_std(outputs)}, min: {tf.math.reduce_min(outputs)}, max: {tf.math.reduce_max(outputs)}")
    #assert tf.math.reduce_std(outputs) > 0.05 and tf.math.reduce_std(outputs) < 0.3
    #assert tf.math.reduce_min(outputs) > -2 and tf.math.reduce_min(outputs) < 0
    #assert tf.math.reduce_max(outputs) < 2 and tf.math.reduce_max(outputs) > 0

def truncated_noise_tests():
    print(f"\n=== {truncated_noise_tests.__name__} ===")
    # Test the truncation sample
    assert tuple(get_truncated_noise(n_samples=10, z_dim=5, truncation=0.7).shape) == (10, 5)
    simple_noise = get_truncated_noise(n_samples=1000, z_dim=10, truncation=0.2)
    assert simple_noise.max() > 0.199 and simple_noise.max() < 2
    assert simple_noise.min() < -0.199 and simple_noise.min() > -0.2
    assert simple_noise.std() > 0.113 and simple_noise.std() < 0.117
    print(f"\n{bcolors.OKGREEN}All test passed!{bcolors.DEFAULT}")

if __name__ == "__main__":
    InitializeGPU()
    SetMemoryLimit(4096)
    truncated_noise_tests()
    NoiseMappingLayersTests()
    InjectNoiseTests()
    AdaINTests()
    MicroStyleGANGeneratorBlockTests()
    #MicroStyleGANGeneratorTests() OOM
    #StyleGANTests() OOM