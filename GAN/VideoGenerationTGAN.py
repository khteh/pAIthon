import matplotlib.pyplot as plt, os, PIL, time, numpy, math, tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from moviepy import ImageSequenceClip
from IPython.display import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Conv3D, MaxPooling3D, LeakyReLU, ReLU, Dropout, Flatten, Dense, BatchNormalization, Reshape, Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.initializers import GlorotUniform, RandomUniform, HeNormal
from tensorflow.keras import losses, optimizers
from tensorflow.keras.regularizers import l2
from utils.Image import ShowImage, CreateGIF
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotGANLossHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

def genSamples(g, n=8):
    '''
    Generate an n by n grid of videos, given a generator g
    '''
    with tf.GradientTape() as tape:
        s = tf.stop_gradient(g(tf.random.uniform((n**2, 100))*2-1)).numpy()

    out = numpy.zeros((3, 16, 64*n, 64*n))
    for j in range(n):
        for k in range(n):
            out[:, :, 64*j:64*(j+1), 64*k:64*(k+1)] = s[j*n+k, :, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (out + 1) / 2 * 255
    out = out.astype(int)
    clip = ImageSequenceClip(list(out), fps=20)
    clip.write_gif('sample.gif', fps=20)

"""
## How to Generate Videos
The first thing to note about video generation is that we are now generating tensors with an added dimension. While conventional image methods work to generate tensors in (H,W,C), we are now generating tensors of size (T,H,W,C).
To solve this problem, TGAN proposed generating temporal dynamics first, then generating images. Gordon and Parde, 2020 have a visual that summarizes the generator's process.

A latent vector vec{z}_c is sampled from a distribution. This vector is fed into some generic G_t and it transforms the vector into a series of latent temporal vectors. 
G_t: z_c -> {z0, z1, ..., zt} From there each temporal vector is joined with vec{z}_c and fed into an image generator G_i. 
With all images created, our last step is to concatenate all of the images to form a video. Under this setup we decompose time and the images.

Today we will be trying to represent the UCF101 dataset. This dataset is composed of 101 action classes. Below is a sample of real examples:
"""
class TemporalGenerator(tf.keras.Model):
    """
    ## The Temporal Generator G_t
    Here we will be implementing our temporal generator. It transforms a vector in R{100} to multiple (16 to be exact) vectors in R{100}. 
    In TGAN they used a series of transposed 1D convolutions, we will discuss the limitations of this choice later.
    """
    def __init__(self):
        super().__init__()
        # Create a sequential model to turn one vector into 16
        self.model = Sequential(
            Input(shape=(100,)),
            Conv1DTranspose(512, kernel_size=1, strides=1, padding="valid", kernel_initializer=GlorotUniform()),
            BatchNormalization(),
            ReLU(),
            Conv1DTranspose(256, kernel_size=4, strides=2, padding="same", kernel_initializer=GlorotUniform()),
            BatchNormalization(),
            ReLU(),
            Conv1DTranspose(128, kernel_size=4, strides=2, padding="same", kernel_initializer=GlorotUniform()),
            BatchNormalization(),
            ReLU(),
            Conv1DTranspose(128, kernel_size=4, strides=2, padding="same", kernel_initializer=GlorotUniform()),
            BatchNormalization(),
            ReLU(),
            Conv1DTranspose(100, kernel_size=4, strides=2, padding="same", activation="tanh", kernel_initializer=GlorotUniform()),
        )

    def call(self, x):
        # reshape x so that it can have convolutions done
        x = x.view(-1, 100, 1)
        # apply the model and flip the
        x = self.model(x).transpose(1, 2)
        return x
    
class VideoGenerator(tf.keras.Model):
    """
    With our vector z_c generated, and our temporal vectors created, it is time to generate our individual images. 
    The first step is to map the two vectors into appropriate sizes to be fed into a transposed 2D convolutional kernel. 
    This is done by a linear transformation with a nonlinearity. Each newly transformed vector is reshaped to a (4,4,256) tensor. 
    In this shape the two sets of vectors are concatenated across the channel dimension.
    After the vectors are transformed, reshaped, and concatenated, it's finally time for us to make the images! 
    TGAN ensues with a generic image generator of multiple transposed 2D convolutions. After enough transposed convolutions, batchnorms, and ReLUs, the final two operations are a transposed convolution to 3 color channels and a tanh activation.
    Our last step is to alter the shape so that the tensor has time, color-channel, height, and width dimensions. We now have a video!
    """
    def __init__(self):
        super().__init__()
        # instantiate the temporal generator
        self.temp = TemporalGenerator()

        # create a transformation for the temporal vectors
        self.fast = Sequential([
            Input(shape=(100,)),
            Dense(256 * 4**2, use_bias=False, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)),
            BatchNormalization(),
            ReLU()
        ])

        # create a transformation for the content vector
        self.slow = Sequential([
            Input(shape=(100,)),
            Dense(256 * 4**2, use_bias=False, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)),
            BatchNormalization(),
            ReLU()
        ])

        # define the image generator
        self.model = Sequential(
            Input(shape=(512,)),
            Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)),
            BatchNormalization(),
            ReLU(),
            Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)),
            BatchNormalization(),
            ReLU(),
            Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)),
            BatchNormalization(),
            ReLU(),
            Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)),
            BatchNormalization(),
            ReLU(),
            Conv2DTranspose(3, kernel_size=3, strides=1, padding="same", activation="tanh", kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01))
        )

    def call(self, x):
        # pass our latent vector through the temporal generator and reshape
        z_fast = self.temp(x)
        z_fast = tf.reshape(z_fast, (-1, 100))

        # transform the content and temporal vectors
        z_fast = tf.reshape(self.fast(z_fast), (-1, 4, 4, 256))
        z_slow = tf.expand_dims(tf.reshape(self.slow(x), (-1, 4, 4, 256)), axis=-1)
        # after z_slow is transformed and expanded we can duplicate it
        z_slow = tf.reshape(tf.concat([z_slow]*16, axis=-1), (-1, 4, 4, 256))

        # concatenate the temporal and content vectors
        z = tf.concat([z_slow, z_fast], axis=-1)

        # transform into image frames
        return tf.transpose(tf.reshape(self.model(z), (-1, 16, 64, 64, 3)), perm=[1,2])
    
class VideoDiscriminator(tf.keras.Model):
    """
    We're no longer operating on images, so now we need to rethink our discriminator. 2D convolutions won't work due to our time dimension, what should we do? 
    TGAN proposes a discriminator composed of a series of 3D convolutions and singular 2D convolution. From one video it produces a single integer.
    Once our discriminator performs inference on some samples the generated integers are then used in the WGAN formulation
    """
    def __init__(self):
        super().__init__()
        self.model3d = Sequential([
            Input(shape=(3,)),
            Conv3D(64, kernel_size=4, padding="same", strides=2, kernel_initializer=HeNormal()),
            LeakyReLU(0.2),
            Conv3D(128, kernel_size=4, padding="same", strides=2, kernel_initializer=HeNormal()),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv3D(256, kernel_size=4, padding="same", strides=2, kernel_initializer=HeNormal()),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv3D(512, kernel_size=4, padding="same", strides=2, kernel_initializer=HeNormal()),
            BatchNormalization(),
            LeakyReLU(0.2)
        ])
        self.conv2d = Conv2D(1, kernel_size=4, strides=1, padding="valid", kernel_initializer=HeNormal())

    def forward(self, x):
        h = self.model3d(x)
        # turn a tensor of R^NxTxCxHxW into R^NxCxHxW
        h = tf.reshape(h, (32, 4, 4, 512))
        h = self.conv2d(h)
        return h