import matplotlib.pyplot as plt, os, PIL, time, numpy, math, tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from moviepy import ImageSequenceClip
from IPython.display import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, ReLU, Dropout, Flatten, Dense, BatchNormalization, Reshape, Conv1DTranspose
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
            Conv1DTranspose(100, 512, kernel_size=1, strides=1, padding="valid"),
            BatchNormalization(),
            ReLU(),
            Conv1DTranspose(512, 256, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            ReLU(),
            Conv1DTranspose(256, 128, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            ReLU(),
            Conv1DTranspose(128, 128, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            ReLU(),
            Conv1DTranspose(128, 100, kernel_size=4, strides=2, padding="same", activation="tanh"),
        )

        # initialize weights according to paper
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == Conv1DTranspose:
            nn.init.xavier_uniform_(m.weight, gain=2**0.5)

    def call(self, x):
        # reshape x so that it can have convolutions done
        x = x.view(-1, 100, 1)
        # apply the model and flip the
        x = self.model(x).transpose(1, 2)
        return x