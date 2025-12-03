import glob, imageio, matplotlib.pyplot as plt, os, time
import numpy, math, tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.layers import Input, Conv2D, Activation, ReLU, Embedding, Dense, BatchNormalization, GroupNormalization, Conv2DTranspose, AveragePooling2D, MaxPool2D, Layer, SpectralNormalization, UpSampling2D, ZeroPadding2D, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.activations import tanh
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.regularizers import l2
from .Pix2PixCityscapeDataGenerator import Pix2PixCityscapeDataGenerator
from utils.GPU import InitializeGPU, UseCPU
from utils.Image import show_tensor_images
from utils.TrainingMetricsPlot import PlotGANLossHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

class ReflectionPad2D(Layer):
  def __init__(self, paddings=(1,1,1,1)):
    super(ReflectionPad2D, self).__init__()
    self.paddings = paddings

  def call(self, input):
    l, r, t, b = self.paddings

    return tf.pad(input, paddings=[[0,0], [t,b], [l,r], [0,0]], mode='REFLECT')

class ResidualBlock(Layer):
    '''
    ResidualBlock Class
    Values
        channels: the number of channels throughout the residual block, a scalar
    '''
    _channels:int = None
    def __init__(self, channels):
        super().__init__()
        self._channels = channels
        self.layers = Sequential([
            ReflectionPad2D(),
            Conv2D(channels, kernel_size=3, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
            GroupNormalization(
                groups=self._channels, # Set groups to the number of channels for Instance Normalization effect
                axis=-1, # channels-last format
                epsilon=1e-5,
                center=True,
                scale=True,
            ),
            ReLU(),
            ReflectionPad2D(),
            Conv2D(channels, kernel_size=3, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
            GroupNormalization(
                groups=self._channels, # Set groups to the number of channels for Instance Normalization effect
                axis=-1, # channels-last format
                epsilon=1e-5,
                center=True,
                scale=True,
            )
        ])
    def call(self, x):
        return x + self.layers(x)
    
class GlobalGenerator(Layer):
    '''
    GlobalGenerator Class
    Implements the global subgenerator (G1) for transferring styles at lower resolutions.
    Even though the global generator is nested inside the local enhancer, you'll still need a separate module for training $G_1$ on its own first.
    Values:
        in_channels: the number of input channels, a scalar
        out_channels: the number of output channels, a scalar
        base_channels: the number of channels in first convolutional layer, a scalar
        fb_blocks: the number of frontend / backend blocks, a scalar
        res_blocks: the number of residual blocks, a scalar
    '''
    def __init__(self, in_channels, out_channels,
                 base_channels=64, fb_blocks=3, res_blocks=9):
        super().__init__()

        # Initial convolutional layer
        g1 = [
            ReflectionPad2D((3,3,3,3)),
            Conv2D(base_channels, kernel_size=7, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
            GroupNormalization(
                groups=base_channels, # Set groups to the number of channels for Instance Normalization effect
                axis=-1, # channels-last format
                epsilon=1e-5,
                center=True,
                scale=True,
            ),
            ReLU()
        ]

        channels = base_channels
        # Frontend blocks
        for _ in range(fb_blocks):
            g1 += [
                Conv2D(2 * channels, kernel_size=3, stride=2, padding="same", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                GroupNormalization(
                    groups=2 * channels, # Set groups to the number of channels for Instance Normalization effect
                    axis=-1, # channels-last format
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                ),
                ReLU()
            ]
            channels *= 2

        # Residual blocks
        for _ in range(res_blocks):
            g1 += [ResidualBlock(channels)]

        # Backend blocks
        for _ in range(fb_blocks):
            g1 += [
                Conv2DTranspose(channels // 2, kernel_size=3, stride=2, padding="same", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                GroupNormalization(
                    groups=channels // 2, # Set groups to the number of channels for Instance Normalization effect
                    axis=-1, # channels-last format
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                ),
                ReLU()
            ]
            channels //= 2

        # Output convolutional layer as its own Sequential since it will be omitted in second training phase
        self.out_layers = Sequential(
            ReflectionPad2D((3,3,3,3)),
            Conv2D(out_channels, kernel_size=7, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
            tanh(),
        )
        self.g1 = Sequential(*g1)

    def call(self, x):
        x = self.g1(x)
        x = self.out_layers(x)
        return x
    
class LocalEnhancer(Layer):
    '''
    LocalEnhancer Class:
    Implements the local enhancer subgenerator (G2) for handling larger scale images.
    The local enhancer uses (a pretrained) G_1 as part of its architecture. The residual connections from the last layers of G_2(F) and G_1(B) are added together and passed through G_2(R) and G_2(B) to synthesize a high-resolution image.
    Because of this, it should reuse the G_1 implementation so that the weights are consistent for the second training phase.
    Values:
        in_channels: the number of input channels, a scalar
        out_channels: the number of output channels, a scalar
        base_channels: the number of channels in first convolutional layer, a scalar
        global_fb_blocks: the number of global generator frontend / backend blocks, a scalar
        global_res_blocks: the number of global generator residual blocks, a scalar
        local_res_blocks: the number of local enhancer residual blocks, a scalar
    '''
    def __init__(self, in_channels, out_channels, base_channels=32, global_fb_blocks=3, global_res_blocks=9, local_res_blocks=3):
        super().__init__()
        global_base_channels = 2 * base_channels

        # Downsampling layer for high-res -> low-res input to g1
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

        # Initialize global generator without its output layers
        self.g1 = GlobalGenerator(
            in_channels, out_channels, base_channels=global_base_channels, fb_blocks=global_fb_blocks, res_blocks=global_res_blocks,
        ).g1

        self.g2 = []

        # Initialize local frontend block
        self.g2.append(
            Sequential(
                # Initial convolutional layer
                ReflectionPad2D((3,3,3,3)),
                Conv2D(base_channels, kernel_size=7, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                GroupNormalization(
                    groups=base_channels, # Set groups to the number of channels for Instance Normalization effect
                    axis=-1, # channels-last format
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                ),
                ReLU(),
                # Frontend block
                Conv2D(2 * base_channels, kernel_size=3, stride=2, padding="same", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                GroupNormalization(
                    groups=2 * base_channels, # Set groups to the number of channels for Instance Normalization effect
                    axis=-1, # channels-last format
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                ),
                ReLU(),
            )
        )

        # Initialize local residual and backend blocks
        self.g2.append(
            Sequential(
                # Residual blocks
                *[ResidualBlock(2 * base_channels) for _ in range(local_res_blocks)],

                # Backend blocks
                Conv2DTranspose(base_channels, kernel_size=3, stride=2, padding="same", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                GroupNormalization(
                    groups=base_channels, # Set groups to the number of channels for Instance Normalization effect
                    axis=-1, # channels-last format
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                ),
                ReLU(),

                # Output convolutional layer
                ReflectionPad2D((3,3,3,3)),
                Conv2D(out_channels, kernel_size=7, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                tanh()
            )
        )

    def call(self, x):
        # Get output from g1_B
        x_g1 = self.downsample(x)
        x_g1 = self.g1(x_g1)

        # Get output from g2_F
        x_g2 = self.g2[0](x)

        # Get final output from g2_B
        return self.g2[1](x_g1 + x_g2)
    
class Discriminator(Layer):
    '''
    Discriminator Class
    Implements the discriminator class for a subdiscriminator, which can be used for all the different scales, just with different argument values.
    Pix2PixHD uses 3 separate subcomponents (subdiscriminators D1, D2 and D3) to generate predictions. They all have the same architectures but D2 and D3 operate on inputs downsampled by 2x and 4x, respectively.
    Each subdiscriminator is a PatchGAN. This implementation will be slightly different than the one you saw in Pix2Pix since the intermediate feature maps will be needed for computing loss.

    Values:
        in_channels: the number of channels in input, a scalar
        base_channels: the number of channels in first convolutional layer, a scalar
        n_layers: the number of convolutional layers, a scalar
    '''
    def __init__(self, in_channels, base_channels=64, n_layers=3):
        super().__init__()

        # Use nn.ModuleList so we can output intermediate values for loss.
        self.layers = []

        # Initial convolutional layer
        self.layers.append(
            Sequential(
                ZeroPadding2D(padding=2), # Adds 2 units of padding on all sides
                Conv2D(base_channels, kernel_size=4, stride=2, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                LeakyReLU(0.2),
            )
        )

        # Downsampling convolutional layers
        channels = base_channels
        for _ in range(1, n_layers):
            prev_channels = channels
            channels = min(2 * channels, 512)
            self.layers.append(
                Sequential(
                    ZeroPadding2D(padding=2), # Adds 2 units of padding on all sides
                    Conv2D(channels, kernel_size=4, stride=2, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                    GroupNormalization(
                        groups=channels, # Set groups to the number of channels for Instance Normalization effect
                        axis=-1, # channels-last format
                        epsilon=1e-5,
                        center=True,
                        scale=True,
                    ),
                    LeakyReLU(0.2),
                )
            )

        # Output convolutional layer
        prev_channels = channels
        channels = min(2 * channels, 512)
        self.layers.append(
            Sequential(
                ZeroPadding2D(padding=2), # Adds 2 units of padding on all sides
                Conv2D(channels, kernel_size=4, stride=1, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                GroupNormalization(
                    groups=channels, # Set groups to the number of channels for Instance Normalization effect
                    axis=-1, # channels-last format
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                ),
                LeakyReLU(0.2),
                ZeroPadding2D(padding=2), # Adds 2 units of padding on all sides
                Conv2D(1, kernel_size=4, stride=1, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))
            )
        )

    def call(self, x):
        outputs = [] # for feature matching loss
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs
    
class MultiscaleDiscriminator(Layer):
    '''
    MultiscaleDiscriminator Class
    The multiscale discriminator in full! This puts together the different subdiscriminator scales.
    Values:
        in_channels: number of input channels to each discriminator, a scalar
        base_channels: number of channels in first convolutional layer, a scalar
        n_layers: number of downsampling layers in each discriminator, a scalar
        n_discriminators: number of discriminators at different scales, a scalar
    '''
    def __init__(self, in_channels, base_channels=64, n_layers=3, n_discriminators=3):
        super().__init__()

        # Initialize all discriminators
        self.discriminators = []
        for _ in range(n_discriminators):
            self.discriminators.append(
                Discriminator(in_channels, base_channels=base_channels, n_layers=n_layers)
            )

        # Downsampling layer to pass inputs between discriminators at different scales
        self.downsample = AveragePooling2D(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        for i, discriminator in enumerate(self.discriminators):
            # Downsample input for subsequent discriminators
            if i != 0:
                x = self.downsample(x)
            outputs.append(discriminator(x))

        # Return list of multiscale discriminator outputs
        return outputs

    @property
    def n_discriminators(self):
        return len(self.discriminators)
    
class Encoder(Layer):
    '''
    Encoder Class
    ## Instance-level Feature Encoder: Adding controllable diversity
    As you already know, the task of generation has more than one possible realistic output. For example, an object of class `road` could be concrete, cobblestone, dirt, etc. To learn this diversity, the authors introduce an encoder E, which takes the original image as input and outputs a feature map (like the feature extractor from Course 2, Week 1). 
    They apply *instance-wise averaging*, averaging the feature vectors across all occurrences of each instance  (so that every pixel corresponding to the same instance has the same feature vector). They then concatenate this instance-level feature embedding with the semantic label and instance boundary maps as input to the generator.
    What's cool is that the encoder E is trained jointly with G1. One huge backprop! When training G2, E is fed a downsampled image and the corresponding output is upsampled to pass into G2.
    To allow for control over different features (e.g. concrete, cobblestone, and dirt) for inference, the authors first use K-means clustering to cluster all the feature vectors for each object class in the training set. You can think of this as a dictionary, mapping each class label to a set of feature vectors (so K centroids, each representing different clusters of features). 
    Now during inference, you can perform a random lookup from this dictionary for each class (e.g. road) in the semantic label map to generate one type of feature (e.g. dirt). To provide greater control, you can select among different feature types for each class to generate diverse feature types and, as a result, multi-modal outputs from the same input.
    Higher values of K increase diversity and potentially decrease fidelity. You've seen this tradeoff between diversity and fidelity before with the truncation trick, and this is just another way to trade-off between them.

    Values:
        in_channels: number of input channels to each discriminator, a scalar
        out_channels: number of channels in output feature map, a scalar
        base_channels: number of channels in first convolutional layer, a scalar
        n_layers: number of downsampling layers, a scalar
    '''
    def __init__(self, in_channels, out_channels, base_channels=16, n_layers=4):
        super().__init__()

        self.out_channels = out_channels
        channels = base_channels

        layers = [
            ReflectionPad2D((3,3,3,3)),
            Conv2D(base_channels, kernel_size=7, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
            GroupNormalization(
                groups=base_channels, # Set groups to the number of channels for Instance Normalization effect
                axis=-1, # channels-last format
                epsilon=1e-5,
                center=True,
                scale=True,
            ),
            ReLU(),
        ]

        # Downsampling layers
        for i in range(n_layers):
            layers += [
                Conv2D(2 * channels, kernel_size=3, stride=2, padding="same", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                GroupNormalization(
                    groups=2 * channels, # Set groups to the number of channels for Instance Normalization effect
                    axis=-1, # channels-last format
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                ),
                ReLU(),
            ]
            channels *= 2

        # Upsampling layers
        for i in range(n_layers):
            layers += [
                Conv2DTranspose(channels // 2, kernel_size=3, stride=2, padding="same", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
                GroupNormalization(
                    groups=channels // 2, # Set groups to the number of channels for Instance Normalization effect
                    axis=-1, # channels-last format
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                ),
                ReLU()
            ]
            channels //= 2

        layers += [
            ReflectionPad2D((3,3,3,3)),
            Conv2D(out_channels, kernel_size=7, padding="valid", kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)),
            tanh()
        ]
        self.layers = Sequential(*layers)

    def instancewise_average_pooling(self, x, inst):
        '''
        Applies instance-wise average pooling.

        Given a feature map of size (b, c, h, w), the mean is computed for each b, c
        across all h, w of the same instance
        '''
        x_mean = tf.zeros_like(x)
        classes = tf.unique(inst, return_inverse=False, return_counts=False) # gather all unique classes present

        for i in classes:
            for b in range(x.size(0)):
                indices = tf.where(inst[b:b+1] == i) # get indices of all positions equal to class i
                for j in range(self.out_channels):
                    x_ins = x[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = tf.reduce_mean(x_ins).expand_as(x_ins)
                    x_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return x_mean

    def call(self, x, inst):
        x = self.layers(x)
        x = self.instancewise_average_pooling(x, inst)
        return x
    
class _VGG19(tf.keras.Model):
    '''
    VGG19 Class
    Wrapper for pretrained VGG19 to output intermediate feature maps
    '''
    def __init__(self):
        super().__init__()
        vgg_features = VGG19(weights='imagenet', include_top=False)
        vgg_features.summary()
        self.f1 = Sequential(*[vgg_features[x] for x in range(2)])
        self.f2 = Sequential(*[vgg_features[x] for x in range(2, 7)])
        self.f3 = Sequential(*[vgg_features[x] for x in range(7, 12)])
        self.f4 = Sequential(*[vgg_features[x] for x in range(12, 21)])
        self.f5 = Sequential(*[vgg_features[x] for x in range(21, 30)])
        for param in self.parameters():
            param.requires_grad = False

    def call(self, x):
        h1 = self.f1(x)
        h2 = self.f2(h1)
        h3 = self.f3(h2)
        h4 = self.f4(h3)
        h5 = self.f5(h4)
        return [h1, h2, h3, h4, h5]

class Loss(Layer):
    '''
    Loss Class
    In addition to the architectural and feature-map enhancements, the authors also incorporate a feature matching loss based on the discriminator. Essentially, they output intermediate feature maps at different resolutions from the discriminator and try to minimize the difference between the real and fake image features.
    The authors found this to stabilize training. In this case, this forces the generator to produce natural statistics at multiple scales. This feature-matching loss is similar to StyleGAN's perceptual loss.
    The authors also report minor improvements in performance when adding perceptual loss.

    Implements composite loss for GauGAN
    Values:
        lambda1: weight for feature matching loss, a float
        lambda2: weight for vgg perceptual loss, a float
        device: 'cuda' or 'cpu' for hardware to use
        norm_weight_to_one: whether to normalize weights to (0, 1], a bool
    '''
    def __init__(self, lambda1=10., lambda2=10., device='cuda', norm_weight_to_one=True):
        super().__init__()
        self.vgg = _VGG19()
        self.vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

        lambda0 = 1.0
        # Keep ratio of composite loss, but scale down max to 1.0
        scale = max(lambda0, lambda1, lambda2) if norm_weight_to_one else 1.0

        self.lambda0 = lambda0 / scale
        self.lambda1 = lambda1 / scale
        self.lambda2 = lambda2 / scale

    def adv_loss(self, discriminator_preds, is_real):
        '''
        Computes adversarial loss from nested list of fakes outputs from discriminator.
        '''
        target = tf.ones_like if is_real else tf.zeros_like

        adv_loss = 0.0
        for preds in discriminator_preds:
            pred = preds[-1]
            adv_loss += F.mse_loss(pred, target(pred))
        return adv_loss

    def fm_loss(self, real_preds, fake_preds):
        '''
        Computes feature matching loss from nested lists of fake and real outputs from discriminator.
        '''
        fm_loss = 0.0
        for real_features, fake_features in zip(real_preds, fake_preds):
            for real_feature, fake_feature in zip(real_features, fake_features):
                fm_loss += MeanAbsoluteError(real_feature, fake_feature) # Original code has .detach() on the real_feature Pytorch tensor
        return fm_loss

    def vgg_loss(self, x_real, x_fake):
        '''
        Computes perceptual loss with VGG network from real and fake images.
        '''
        vgg_real = self.vgg(x_real)
        vgg_fake = self.vgg(x_fake)

        vgg_loss = 0.0
        for real, fake, weight in zip(vgg_real, vgg_fake, self.vgg_weights):
            vgg_loss += weight * MeanAbsoluteError(real, fake) # Original code has .detach() on the real Pytorch tensor
        return vgg_loss

    def call(self, x_real, label_map, instance_map, boundary_map, encoder, generator, discriminator):
        '''
        Function that computes the forward pass and total loss for generator and discriminator.
        '''
        feature_map = encoder(x_real, instance_map)
        x_fake = generator(tf.concat((label_map, boundary_map, feature_map), axis=1))

        # Get necessary outputs for loss/backprop for both generator and discriminator
        fake_preds_for_g = discriminator(tf.concat((label_map, boundary_map, x_fake), axis=1))
        fake_preds_for_d = discriminator(tf.concat((label_map, boundary_map, x_fake), axis=1)) # Original code has .detach() on the x_fake Pytorch tensor
        real_preds_for_d = discriminator(tf.concat((label_map, boundary_map, x_real), axis=1)) # Original code has .detach() on the x_real Pytorch tensor

        g_loss = (
            self.lambda0 * self.adv_loss(fake_preds_for_g, True) + \
            self.lambda1 * self.fm_loss(real_preds_for_d, fake_preds_for_g) / discriminator.n_discriminators + \
            self.lambda2 * self.vgg_loss(x_fake, x_real)
        )
        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) + \
            self.adv_loss(fake_preds_for_d, False)
        )
        return g_loss, d_loss, x_fake.detach() # Original code has .detach() on the x_fake Pytorch tensor

# https://www.tensorflow.org/datasets/catalog/cityscapes

n_classes = 35                  # total number of object classes
rgb_channels = n_features = 3
device = 'cuda'
train_dir = ['data']
epochs = 200                    # total number of train epochs
decay_after = 100               # number of epochs with constant lr
lr = 0.0002
betas = (0.5, 0.999)

def lr_lambda(epoch):
    ''' Function for scheduling learning '''
    return 1. if epoch < decay_after else 1 - float(epoch - decay_after) / (epochs - decay_after)

loss_fn = Loss(device=device)

## Phase 1: Low Resolution (1024 x 512)
#dataloader1 = DataLoader(
#    CityscapesDataset(train_dir, target_width=1024, n_classes=n_classes),
#    collate_fn=CityscapesDataset.collate_fn, batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
#)
# def __init__(self, paths, target_width=1024, n_classes=35):
dataloader1 = Pix2PixCityscapeDataGenerator(f"data/cityscape/", target_width=1024, n_classes=n_classes)
encoder = Encoder(rgb_channels, n_features)
generator1 = GlobalGenerator(n_classes + n_features + 1, rgb_channels)
discriminator1 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels, n_discriminators=2)

g1_optimizer = Adam(list(generator1.parameters()) + list(encoder.parameters()), learning_rate=lr_lambda, beta_1 = betas[0], beta_2 = betas[1])
d1_optimizer = Adam(list(discriminator1.parameters()), learning_rate=lr_lambda, beta_1 = betas[0], beta_2 = betas[1])

## Phase 2: High Resolution (2048 x 1024)
#dataloader2 = DataLoader(
#    CityscapesDataset(train_dir, target_width=2048, n_classes=n_classes),
#    collate_fn=CityscapesDataset.collate_fn, batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
#)
dataloader2 = Pix2PixCityscapeDataGenerator(f"data/cityscape/", target_width=2048, n_classes=n_classes)
generator2 = LocalEnhancer(n_classes + n_features + 1, rgb_channels)
discriminator2 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels)

g2_optimizer = Adam(list(generator2.parameters()) + list(encoder.parameters()), learning_rate=lr_lambda, beta_1 = betas[0], beta_2 = betas[1])
d2_optimizer = Adam(list(discriminator2.parameters()), learning_rate=lr_lambda, beta_1 = betas[0], beta_2 = betas[1])

def train(dataloader, models, optimizers):
    encoder, generator, discriminator = models
    g_optimizer, d_optimizer = optimizers

    cur_step = 0
    display_step = 100

    mean_g_loss = 0.0
    mean_d_loss = 0.0

    for epoch in tqdm(range(epochs)):
        start = time.time()
        for (x_real, labels, insts, bounds) in tqdm(dataloader, position=0):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                g_loss, d_loss, x_fake = loss_fn(
                    x_real, labels, insts, bounds, encoder, generator, discriminator
                )
            # Compute gradients
            g_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
            d_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)

            # Apply gradients to update weights to model
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step
            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'.format(cur_step, mean_g_loss, mean_d_loss))
                show_tensor_images(x_fake.to(x_real.dtype))
                show_tensor_images(x_real)
                mean_g_loss = 0.0
                mean_d_loss = 0.0
            cur_step += 1
        print(f"Epoch {epoch + 1}: {time.time()-start}s")

# Phase 1: Low Resolution
#######################################################################
train(
    dataloader1,
    [encoder, generator1, discriminator1],
    [g1_optimizer, d1_optimizer]
)

# Phase 2: High Resolution
#######################################################################
# Update global generator in local enhancer with trained
generator2.g1 = generator1.g1

# Freeze encoder and wrap to support high-resolution inputs/outputs
def freeze(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    def _resize(x, scale_factor:float):
        new_height = tf.cast(tf.shape(x)[1] * scale_factor, tf.int32)
        new_width = tf.cast(tf.shape(x)[2] * scale_factor, tf.int32)
        return [new_height, new_width]        

    @tf.function
    def forward(x, inst):
        x = tf.image.resize(x, size=_resize(x, 0.5), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        inst = tf.image.resize(x, size=_resize(x, 0.5), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        feat = encoder(x, inst.int())
        return tf.image.resize(feat, size=_resize(x, 2.0), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    
    return forward

train(
    dataloader2,
    [freeze(encoder), generator2, discriminator2],
    [g2_optimizer, d2_optimizer]
)

## Inference with Pix2PixHD
# Recall that in inference time, the encoder feature maps from training are saved and clustered with K-means by object class.

# Encode features by class label
features = {}
for (x, _, inst, _) in tqdm(dataloader2):
    area = inst.size(2) * inst.size(3)

    # Get pooled feature map
    # Perform an operation without gradient tracking (similar to torch.no_grad)
    with tf.GradientTape() as tape:
        feature_map = encoder(tf.stop_gradient(x), tf.stop_gradient(inst))

    for i in tf.unique(inst):
        label = i if i < 1000 else i // 1000
        label = int(label.flatten(0).item())

        # All indices should have same feature per class from pooling
        idx = tf.where(inst == i, as_tuple=False)
        n_inst = idx.size(0)
        idx = idx[0, :]

        # Retrieve corresponding encoded feature
        feature = feature_map[idx[0], :, idx[2], idx[3]].unsqueeze(0)

        # Compute rate of feature appearance (in official code, they compute per block)
        block_size = 32
        rate_per_block = 32 * n_inst / area
        rate = tf.ones((1, 1), device=device).to(feature.dtype) * rate_per_block

        feature = tf.concat((feature, rate), dim=1)
        if label in features.keys():
            features[label] = tf.concat((features[label], feature), dim=0)
        else:
            features[label] = feature


# Cluster features by class label
k = 10
centroids = {}
for label in range(n_classes):
    if label in features.keys():
        feature = features[label]

        # Thresholding by 0.5 isn't mentioned in the paper, but is present in the
        # official code repository, probably so that only frequent features are clustered
        feature = feature[feature[:, -1] > 0.5, :-1].cpu().numpy()

        if feature.shape[0]:
            n_clusters = min(feature.shape[0], k)
            kmeans = KMeans(n_clusters=n_clusters).fit(feature)
            centroids[label] = kmeans.cluster_centers_

def infer(label_map, instance_map, boundary_map):
    # Sample feature vector centroids
    b, _, h, w = label_map.shape
    feature_map = tf.zeros((b, n_features, h, w), device=device).to(label_map.dtype)

    for i in tf.unique(instance_map):
        label = i if i < 1000 else i // 1000
        label = int(label.flatten(0).item())

        if label in centroids.keys():
            centroid_idx = rng.integers(low=0, high=centroids[label].shape[0] - 1)
            idx = tf.where(instance_map == int(i), as_tuple=False)

            feature = tf.convert_to_tensor(centroids[label][centroid_idx, :]).to(device)
            feature_map[idx[:, 0], :, idx[:, 2], idx[:, 3]] = feature

    with tf.GradientTape() as tape:
        x_fake = generator2(tf.concat((label_map, boundary_map, feature_map), dim=1))
    return x_fake

for x, labels, insts, bounds in dataloader2:
    x_fake = infer(labels.to(device), insts.to(device), bounds.to(device))
    show_tensor_images(x_fake.to(x.dtype))
    show_tensor_images(x)
    break