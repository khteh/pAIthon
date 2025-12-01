import glob, imageio, matplotlib.pyplot as plt, os, time
import numpy, math, tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from utils.Image import CreateGIF, ShowImage
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Activation, ReLU, Embedding, Dense, BatchNormalization, Reshape, AveragePooling2D, Layer, SpectralNormalization, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.regularizers import l2
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotGANLossHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

def orthogonal_regularization(weight):
    '''
    Function for computing the orthogonal regularization term for a given weight matrix.
    '''
    weight = weight.flatten(1)
    return tf.norm(
        tf.dot(weight, weight) * (tf.ones_like(weight) - tf.eye(weight.shape[0]))
    )

class ClassConditionalBatchNorm2d(Layer):
    '''
    Class-conditional Batch Normalization

    Recall that batch norm aims to normalize activation statistics to a standard gaussian distribution (via an exponential moving average of minibatch mean and variances) but also applies trainable parameters, gamma and beta, to invert this operation if the model sees fit:
    y = ((x - miu^) / (alpha^ + epsilon)) * gamma + beta

    BigGAN injects class-conditional information by parameterizing gamma and beta as linear transformations of the class embedding, c. Recall that BigGAN also concatenates c with z skip connections (denoted [c, z]), so
    gamma := Wgamma.T * [c,z]
    beta := Wbeta.T * [c,z]

    The idea is actually very similar to the adaptive instance normalization (AdaIN) module that you implemented in the StyleGAN notebook.

    Values:
    in_channels: the dimension of the class embedding (c) + noise vector (z), a scalar
    out_channels: the dimension of the activation tensor to be normalized, a scalar
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self._input = Input(shape=(in_channels, ))
        self.bn = BatchNormalization()
        self.class_scale_transform = SpectralNormalization(Dense(out_channels, use_bias=False, kernel_initializer=Orthogonal()))
        self.class_shift_transform = SpectralNormalization(Dense(out_channels, use_bias=False, kernel_initializer=Orthogonal()))

    def call(self, x, y):
        #x = self._input(x)
        normalized_image = self.bn(x)
        class_scale = (1 + self.class_scale_transform(y))[:, :, None, None]
        class_shift = self.class_shift_transform(y)[:, :, None, None]
        transformed_image = class_scale * normalized_image + class_shift
        return transformed_image

class AttentionBlock(Layer):
    '''
    Self-attention has been a successful technique in helping models learn arbitrary, long-term dependencies. [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318) (Zhang et al. 2018) first introduced the self-attention mechanism into the GAN architecture. BigGAN augments its residual blocks with these attention blocks.
    **A Quick Primer on Self-Attention**
    Self-attention is just **scaled dot product attention**. Given a sequence S (with images, S is just the image flattened across its height and width), the model learns mappings to query (Q), key (K), and value (V) matrices:

        Q := W_q.T * S
        K := W_k.T * S
        V := W_v.T * S

    where W_q, W_k, and W_v are learned parameters. The subsequent self-attention mechanism is then computed as Attention(Q, K, V) = softmax((Q*K.T) / sqrt(d_k)) * V
    where d_k is the dimensionality of the Q, K matrices (SA-GAN and BigGAN both omit this term). Intuitively, you can think of the *query* matrix as containing the representations of each position with respect to itself and the *key* matrix as containing the representations of each position with respect to the others. 
    How important two positions are to each other is measured by dot product as Q * K.T, hence **dot product attention**. A softmax is applied to convert these relative importances to a probability distribution over all positions.
    Intuitively, the *value* matrix provides the importance weighting of the attention at each position, hence **scaled dot product attention**. Relevant positions should be assigned larger weight and irrelevant ones should be assigned smaller weight.

    Values:
    channels: number of channels in input
    '''
    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.theta = SpectralNormalization(Conv2D(channels // 8, kernel_size=1, padding="valid", use_bias=False, kernel_initializer=Orthogonal()))
        self.phi = SpectralNormalization(Conv2D(channels // 8, kernel_size=1, padding="valid", use_bias=False, kernel_initializer=Orthogonal()))
        self.g = SpectralNormalization(Conv2D(channels // 2, kernel_size=1, padding="valid", use_bias=False, kernel_initializer=Orthogonal()))
        self.o = SpectralNormalization(Conv2D(channels, kernel_size=1, padding="valid", use_bias=False, kernel_initializer=Orthogonal()))
        self.gamma = tf.Variable(0., requires_grad=True)

    def call(self, x):
        spatial_size = x.shape[2] * x.shape[3]

        # Apply convolutions to get query (theta), key (phi), and value (g) transforms
        theta = self.theta(x)
        phi = tf.nn.max_pool2d(self.phi(x), kernel_size=2)
        g = tf.nn.max_pool2d(self.g(x), kernel_size=2)

        # Reshape spatial size for self-attention
        theta = theta.view(-1, self.channels // 8, spatial_size)
        phi = phi.view(-1, self.channels // 8, spatial_size // 4)
        g = g.view(-1, self.channels // 2, spatial_size // 4)

        # Compute dot product attention with query (theta) and key (phi) matrices
        beta = tf.nn.softmax(tf.matmul(theta.transpose(1, 2), phi), dim=-1)

        # Compute scaled dot product attention with value (g) and attention (beta) matrices
        o = self.o(tf.matmul(g, beta.transpose(1, 2)).view(-1, self.channels // 2, x.shape[2], x.shape[3]))

        # Apply gain and residual
        return self.gamma * o + x
    
class GResidualBlock(Layer):
    '''
    ### Generator Residual Block

    As with many state-of-the-art computer vision models, BigGAN employs skip connections in the form of residual blocks to map random noise to a fake image. You can think of BigGAN residual blocks as having 3 steps. Given input $x$ and class embedding y:
    1. h := bn-relu-upsample-conv(x, y)
    2. h := bn-relu-conv(h, y)
    3. x := upsample-conv(x)

    after which you can apply a residual connection and return h + x.

    Values:
    c_dim: the dimension of conditional vector [c, z], a scalar
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    '''
    def __init__(self, c_dim, in_channels, out_channels):
        super().__init__()

        self.conv1 = SpectralNormalization(Conv2D(out_channels, kernel_size=3, padding="same", kernel_initializer=Orthogonal()))
        self.conv2 = SpectralNormalization(Conv2D(out_channels, kernel_size=3, padding="same", kernel_initializer=Orthogonal()))

        self.bn1 = ClassConditionalBatchNorm2d(c_dim, in_channels)
        self.bn2 = ClassConditionalBatchNorm2d(c_dim, out_channels)

        self.activation = ReLU()
        self.upsample_fn = UpSampling2D(size=2)     # upsample occurs in every gblock

        self.mixin = (in_channels != out_channels)
        if self.mixin:
            self.conv_mixin = SpectralNormalization(Conv2D(out_channels, kernel_size=1, padding="valid", kernel_initializer=Orthogonal()))

    def call(self, x, y):
        # h := upsample(x, y)
        h = self.bn1(x, y)
        h = self.activation(h)
        h = self.upsample_fn(h)
        h = self.conv1(h)

        # h := conv(h, y)
        h = self.bn2(h, y)
        h = self.activation(h)
        h = self.conv2(h)

        # x := upsample(x)
        x = self.upsample_fn(x)
        if self.mixin:
            x = self.conv_mixin(x)
        return h + x
    
class Generator():
    '''
    Generator Class
    Values:
    z_dim: the dimension of random noise sampled, a scalar
    shared_dim: the dimension of shared class embeddings, a scalar
    base_channels: the number of base channels, a scalar
    bottom_width: the height/width of image before it gets upsampled, a scalar
    n_classes: the number of image classes, a scalar
    '''
    _base_channels: int = None
    _bottom_width:int = None
    _z_dim:int = None
    _z_chunk_size: int = None
    _shared_dim:int = None
    shared_emb = None
    _classes:int = None
    _g_blocks = []
    _proj_z = None
    _proj_o = None
    model = None
    optimizer = None
    def __init__(self, base_channels=96, bottom_width=4, z_dim=120, shared_dim=128, n_classes=1000):
        #super().__init__()
        self._base_channels = base_channels
        self._bottom_width = bottom_width
        self._z_dim = z_dim
        self._shared_dim = shared_dim
        self._classes = n_classes
 
        n_chunks = 6    # 5 (generator blocks) + 1 (generator input)
        self._z_chunk_size = z_dim // n_chunks
 
        # No spectral normalization on embeddings, which authors observe to cripple the generator
        self.shared_emb = Embedding(n_classes, shared_dim)
 
        self._proj_z = Dense(16 * base_channels * bottom_width ** 2, kernel_initializer=Orthogonal())

        # Can't use one big nn.Sequential since we are adding class+noise at each block
        self._g_blocks.append([
                GResidualBlock(shared_dim + self._z_chunk_size, 16 * base_channels, 16 * base_channels),
                AttentionBlock(16 * base_channels),
            ])
        self._g_blocks.append([
                GResidualBlock(shared_dim + self._z_chunk_size, 16 * base_channels, 8 * base_channels),
                AttentionBlock(8 * base_channels),
            ])
        self._g_blocks.append([
                GResidualBlock(shared_dim + self._z_chunk_size, 8 * base_channels, 4 * base_channels),
                AttentionBlock(4 * base_channels),
            ])
        self._g_blocks.append([
                GResidualBlock(shared_dim + self._z_chunk_size, 4 * base_channels, 2 * base_channels),
                AttentionBlock(2 * base_channels),
            ])
        self._g_blocks.append([
                GResidualBlock(shared_dim + self._z_chunk_size, 2 * base_channels, base_channels),
                AttentionBlock(base_channels),
            ])
        self._proj_o = Sequential([
            BatchNormalization(base_channels),
            ReLU(),
            SpectralNormalization(Conv2D(3, kernel_size=1, padding="valid", kernel_initializer=Orthogonal())),
            Activation("tanh")
        ])
        self.optimizer = Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.999, epsilon=1e-6) # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.

    def forward(self, z, y):
        '''
        z: random noise with size self.z_dim
        y: class embeddings with size self.shared_dim
            = NOTE =
            y should be class embeddings from self.shared_emb, not the raw class labels
        '''
        # Chunk z and concatenate to shared class embeddings
        zs = tf.split(z, self._z_chunk_size, axis=1)
        z = zs[0]
        ys = [tf.concat([y, z], axis=1) for z in zs[1:]]
 
        # Project noise and reshape to feed through generator blocks
        h = self._proj_z(z)
        h = tf.reshape(h, [h.shape[0], -1, self._bottom_width, self._bottom_width])
 
        # Feed through generator blocks
        for idx, g_block in enumerate(self._g_blocks):
            h = g_block[0](h, ys[idx])
            h = g_block[1](h)
 
        # Project to 3 RGB channels with tanh to map values to [-1, 1]
        return self._proj_o(h)

class DResidualBlock(Layer):
    '''
    Discriminator residual block, which is simpler than the generator's. Note that the last residual block does not apply downsampling.
    1. h := relu-conv-relu-downsample(x)
    2. x := conv-downsample(x)

    In the official BigGAN implementation, the architecture is slightly different for the first discriminator residual block, since it handles the raw image as input:
    1. h := conv-relu-downsample(x)
    2. x := downsample-conv(x)

    After these two steps, you can return the residual connection h + x. You might notice that there is no class information in these residual blocks. 
    The authors inject class-conditional information after the final hidden layer (and before the output layer) via channel-wise dot product.

    Values:
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    downsample: whether to apply downsampling
    use_preactivation: whether to apply an activation function before the first convolution
    '''

    def __init__(self, in_channels, out_channels, downsample=True, use_preactivation=False):
        super().__init__()

        self.conv1 = SpectralNormalization(Conv2D(out_channels, kernel_size=3, padding="same", kernel_initializer=Orthogonal()))
        self.conv2 = SpectralNormalization(Conv2D(out_channels, kernel_size=3, padding="same", kernel_initializer=Orthogonal()))

        self.activation = ReLU()
        self.use_preactivation = use_preactivation  # apply preactivation in all except first dblock

        self.downsample = downsample    # downsample occurs in all except last dblock
        if downsample:
            self.downsample_fn = AveragePooling2D(2)
        self.mixin = (in_channels != out_channels) or downsample
        if self.mixin:
            self.conv_mixin = SpectralNormalization(Conv2D(out_channels, kernel_size=1, padding="valid", kernel_initializer=Orthogonal()))

    def _residual(self, x):
        if self.use_preactivation:
            if self.mixin:
                x = self.conv_mixin(x)
            if self.downsample:
                x = self.downsample_fn(x)
        else:
            if self.downsample:
                x = self.downsample_fn(x)
            if self.mixin:
                x = self.conv_mixin(x)
        return x

    def forward(self, x):
        # Apply preactivation if applicable
        if self.use_preactivation:
            h = tf.nn.relu(x)
        else:
            h = x

        h = self.conv1(h)
        h = self.activation(h)
        if self.downsample:
            h = self.downsample_fn(h)

        return h + self._residual(x)
    
class Discriminator():
    '''
    Discriminator Class
    Values:
    base_channels: the number of base channels, a scalar
    n_classes: the number of image classes, a scalar
    '''
    _base_channels:int = None
    _classes:int = None
    shared_emb = None
    _d_blocks = None
    _proj_o = None
    model = None
    optimizer = None
    def __init__(self, base_channels=96, n_classes=1000):
        #super().__init__()
        self._base_channels = base_channels
        self._classes = n_classes

        # For adding class-conditional evidence
        self.shared_emb = SpectralNormalization(Embedding(n_classes, 16 * base_channels))

        self._d_blocks = Sequential([
            DResidualBlock(3, base_channels, downsample=True, use_preactivation=False),
            AttentionBlock(base_channels),

            DResidualBlock(base_channels, 2 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(2 * base_channels),

            DResidualBlock(2 * base_channels, 4 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(4 * base_channels),

            DResidualBlock(4 * base_channels, 8 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(8 * base_channels),

            DResidualBlock(8 * base_channels, 16 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(16 * base_channels),

            DResidualBlock(16 * base_channels, 16 * base_channels, downsample=False, use_preactivation=True),
            AttentionBlock(16 * base_channels),

            ReLU(),
        ])
        self._proj_o = SpectralNormalization(Dense(1, kernel_initializer=Orthogonal()))
        self.optimizer = Adam(learning_rate=4e-4, beta_1=0.0, beta_2=0.999, epsilon=1e-6)

    def forward(self, x, y=None):
        h = self._d_blocks(x)
        h = tf.math.reduce_sum(h, axis=[2, 3])

        # Class-unconditional output
        uncond_out = self.proj_o(h)
        if y is None:
            return uncond_out

        # Class-conditional output
        cond_out = tf.math.reduce_sum(self.shared_emb(y) * h, axis=1, keepdim=True)
        return uncond_out + cond_out
    
class BigGAN():
    _generator:Generator = None
    _discrinimator:Discriminator = None
    _checkpoint: tf.train.Checkpoint = None
    n_classes:int = None
    def __init__(self, base_channels, z_dim, n_classes, shared_dim):
        self._classes = n_classes
        self._generator = Generator(base_channels=base_channels, bottom_width=4, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes)
        self._discriminator = Discriminator(base_channels=base_channels, n_classes=n_classes)
        """
        self._checkpoint = tf.train.Checkpoint(generator_optimizer = self._generator.optimizer,
                                        discriminator_optimizer = self._discriminator.optimizer,
                                        generator = self._generator.model,
                                        discriminator = self._discriminator.model)
        """
    def SampleForwardPass(self):
        batch_size = self._classes
        z = tf.random.normal((batch_size, z_dim))                 # Generate random noise (z)
        y = tf.range(start=0, limit=self._classes, dtype=tf.int64)# Generate a batch of labels (y), one for each class
        y_emb = self._generator.shared_emb(y)                     # Retrieve class embeddings (y_emb) from generator

        x_gen = self._generator.forward(z, y_emb)                 # Generate fake images from z and y_emb
        score = self._discriminator.forward(x_gen, y)             # Generate classification for fake images
        print(f"score: {score}")

if __name__ == "__main__":
    # Initialize models
    base_channels = 96
    z_dim = 120
    n_classes = 5   # 5 classes is used instead of the original 1000, for efficiency
    shared_dim = 128
    InitializeGPU()
    biggan = BigGAN(base_channels, z_dim, n_classes, shared_dim)
    biggan.SampleForwardPass()