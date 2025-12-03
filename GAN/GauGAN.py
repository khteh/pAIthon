import glob, imageio, matplotlib.pyplot as plt, os, time, getpass
import numpy, math, tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from utils.Image import CreateGIF, ShowImage
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Activation, ReLU, Embedding, Flatten, Dense, BatchNormalization, GroupNormalization, AveragePooling2D, MaxPool2D, Layer, SpectralNormalization, UpSampling2D, LeakyReLU, ZeroPadding2D
from tensorflow.keras.activations import tanh
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.regularizers import l2
from .GauGANCityscapeDataGenerator import GauGANCityscapeDataGenerator
from utils.Image import show_tensor_images
from utils.GPU import InitializeGPU, UseCPU
from utils.TrainingMetricsPlot import PlotGANLossHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

class SPADE(Layer):
    '''
    SPADE Class
    With vanilla batch norm, these denormalization parameters are spatially invariant - that is, the same values are applied to every position in the input activation. 
    As you may imagine, this could be limiting for the model. Oftentimes it's conducive for the model to learn denormalization parameters for each position.
    The authors address this with **SPatially Adaptive DEnormalization (SPADE)**. They compute denormalization parameters gamma and beta by convolving the input segmentation masks and apply these elementwise.
    Note: the authors use spectral norm in all convolutional layers in the generator and discriminator of GauGAN, but the official code omits spectral norm for SPADE layers.

    Values:
        channels: the number of channels in the input, a scalar
        cond_channels: the number of channels in conditional input (one-hot semantic labels), a scalar
    '''
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.batchnorm = BatchNormalization()
        self.spade = Sequential([
            #Input(shape=(cond_channels,)),
            Conv2D(channels, kernel_size=3, padding="same"),
            ReLU(),
            Conv2D(2 * channels, kernel_size=3, padding="same"),
        ])
    def call(self, x, seg):
        # Apply normalization
        x = self.batchnorm(x)

        # Compute denormalization
        seg = tf.image.resize(seg, size=x.shape[-2:], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        gamma, beta = tf.split(self.spade(seg), 2, axis=1)

        # Apply denormalization
        x = x * (1 + gamma) + beta
        return x

class ResidualBlock(Layer):
    '''
    ResidualBlock Class
    Residual Blocks with SPADE normalization. This implementation will be a bit different to accomodate for the extra semantic label map input. For a refresher on residual blocks, please take a look [here](https://paperswithcode.com/method/residual-block).    
    Values:
        in_channels: the number of input channels, a scalar
        out_channels: the number of output channels, a scalar
        cond_channels: the number of channels in conditional input in spade layer, a scalar
    '''
    def __init__(self, in_channels, out_channels, cond_channels):
        super().__init__()
        hid_channels = min(in_channels, out_channels)
        self.proj = in_channels != out_channels
        if self.proj:
            self.norm0 = SPADE(in_channels, cond_channels)
            self.conv0 = SpectralNormalization(
                Conv2D(out_channels, kernel_size=1, use_bias=False)
            )
        self.activation = LeakyReLU(0.2)
        self.norm1 = SPADE(in_channels, cond_channels)
        self.norm2 = SPADE(hid_channels, cond_channels)
        self.conv1 = SpectralNormalization(
            Conv2D(hid_channels, kernel_size=3, padding="same")
        )
        self.conv2 = SpectralNormalization(
            Conv2D(out_channels, kernel_size=3, padding="same")
        )
    def call(self, x, seg):
        dx = self.norm1(x, seg)
        dx = self.activation(dx)
        dx = self.conv1(dx)
        dx = self.norm2(dx, seg)
        dx = self.activation(dx)
        dx = self.conv2(dx)
        # Learn skip connection if in_channels != out_channels
        if self.proj:
            x = self.norm0(x, seg)
            x = self.conv0(x)
        return x + dx

class Encoder(Layer):
    '''
    Encoder Class
    GauGAN's encoder serves a different purpose than Pix2PixHD's. Instead of learning feature maps to be fed as input to the generator, GauGAN's encoder encodes the original image into a mean and standard deviation from which to sample noise, which is given to the generator. This same technique of encoding to a mean and standard devation is used in variational autoencoders (VAEs)

    Values:
        spatial_size: tuple specifying (height, width) of full size image, a tuple
        z_dim: number of dimensions of latent noise vector (z), a scalar
        n_downsample: number of downsampling blocks in the encoder, a scalar
        base_channels: number of channels in the last hidden layer, a scalar
    '''
    max_channels = 512
    def __init__(self, spatial_size, z_dim=256, n_downsample=6, base_channels=64):
        super().__init__()
        layers = []
        channels = base_channels
        for i in range(n_downsample):
            in_channels = 3 if i == 0 else channels
            out_channels = 2 * z_dim if i < n_downsample else max(self.max_channels, channels * 2)
            layers += [
                SpectralNormalization(
                    Conv2D(out_channels, strides=2, kernel_size=3, padding="same")
                ),
                GroupNormalization(out_channels),
                LeakyReLU(0.2)
            ]
            channels = out_channels

        h, w = spatial_size[0] // 2 ** n_downsample, spatial_size[1] // 2 ** n_downsample
        layers += [
            Flatten(),
            Dense(2 * z_dim)
        ]
        self.layers = Sequential([*layers])

    def call(self, x):
        return tf.split(self.layers(x), 2, dim=1)

class Generator(Layer):
    '''
    Generator Class
    The GauGAN generator is actually very different from previous image-to-image translation generators. 
    Because information from the semantic label map is injected at each batch normalization layer, the generator is able to just take random noise $z$ as input. 
    This noise is reshaped and upsampled to the target image size.

    Values:
        n_classes: the number of object classes in the dataset, a scalar
        bottom_width: the downsampled spatial size of the image, a scalar
        z_dim: the number of dimensions the z noise vector has, a scalar
        base_channels: the number of channels in last hidden layer, a scalar
        n_upsample: the number of upsampling operations to apply, a scalar
    '''
    max_channels = 1024
    def __init__(self, n_classes, spatial_size, z_dim=256, base_channels=64, n_upsample=6):
        super().__init__()
        h, w = spatial_size[0] // 2 ** n_upsample, spatial_size[1] // 2 ** n_upsample
        self.proj_z = Dense(self.max_channels * h * w)
        self.reshape = lambda x: tf.reshape(x, (-1, self.max_channels, h, w))
        self.upsample = UpSampling2D(size=2)
        self.res_blocks = []
        for i in reversed(range(n_upsample)):
            in_channels = min(self.max_channels, base_channels * 2 ** (i+1))
            out_channels = min(self.max_channels, base_channels * 2 ** i)
            self.res_blocks.append(ResidualBlock(in_channels, out_channels, n_classes))

        self.proj_o = Sequential([
            #Input(shape=(base_channels,)),
            Conv2D(3, kernel_size=3, padding="same", activation="tanh"),
        ])
    def call(self, z, seg):
        h = self.proj_z(z)
        h = self.reshape(h)
        for res_block in self.res_blocks:
            h = res_block(h, seg)
            h = self.upsample(h)
        h = self.proj_o(h)
        return h

class PatchGANDiscriminator(Layer):
    '''
    PatchGANDiscriminator Class
    Implements the discriminator class for a subdiscriminator, which can be used for all the different scales, just with different argument values.
    The architecture of the discriminator follows the one used in Pix2PixHD, which uses a multi-scale design with the InstanceNorm. 
    The only difference here is that they apply spectral normalization to all convolutional layers. 
    GauGAN's discriminator also takes as input the image concatenated with the semantic label map (no instance boundary map as in Pix2PixHD).

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
            Sequential([
                #Input(shape=(in_channels,)),
                ZeroPadding2D(padding=2), # Adds 2 units of padding on all sides
                SpectralNormalization(
                    Conv2D(base_channels, kernel_size=4, strides=2, padding="valid")
                ),
                LeakyReLU(0.2)
            ])
        )

        # Downsampling convolutional layers
        channels = base_channels
        for _ in range(1, n_layers):
            prev_channels = channels
            channels = min(2 * channels, 512)
            self.layers.append(
                Sequential([
                    #Input(shape=(prev_channels,)),
                    ZeroPadding2D(padding=2), # Adds 2 units of padding on all sides
                    SpectralNormalization(
                        Conv2D(channels, kernel_size=4, strides=2, padding="valid")
                    ),
                    GroupNormalization(channels),
                    LeakyReLU(0.2)
                ])
            )

        # Output convolutional layer
        prev_channels = channels
        channels = min(2 * channels, 512)
        self.layers.append(
            Sequential([
                #Input(shape=(prev_channels,)),
                ZeroPadding2D(padding=2), # Adds 2 units of padding on all sides
                SpectralNormalization(
                    Conv2D(channels, kernel_size=4, strides=1, padding="valid")
                ),
                GroupNormalization(channels),
                LeakyReLU(0.2),
                ZeroPadding2D(padding=2), # Adds 2 units of padding on all sides
                SpectralNormalization(
                    Conv2D(1, kernel_size=4, strides=1, padding="valid")
                ),
            ])
        )
    def call(self, x):
        outputs = [] # for feature matching loss
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs
    
class Discriminator(Layer):
    '''
    Discriminator Class
    The multiscale discriminator in full which puts together the different subdiscriminator scales.

    Values:
        in_channels: number of input channels to each discriminator, a scalar
        base_channels: number of channels in last hidden layer, a scalar
        n_layers: number of downsampling layers in each discriminator, a scalar
        n_discriminators: number of discriminators at different scales, a scalar
    '''
    def __init__(self, in_channels, base_channels=64, n_layers=3, n_discriminators=3):
        super().__init__()

        # Initialize all discriminators
        self.discriminators = []
        for _ in range(n_discriminators):
            self.discriminators.append(
                PatchGANDiscriminator(in_channels, base_channels=base_channels, n_layers=n_layers)
            )

        # Downsampling layer to pass inputs between discriminators at different scales
        self.downsample = AveragePooling2D(3, strides=2, padding="same")

    def call(self, x):
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
    
class GauGAN(tf.keras.Model):
    '''
    GauGAN Class
    Since the encoder outputs mean and log-variance values to sample random noise from, this implementation will use the 'reparameterization trick' to allow gradient flow to the encoder.
    This trick samples from N(0,I) and applies shift and scale (miu, gamma) as opposed to sampling directly from N(miu, gamma^2 * I).

    Values:
        n_classes: number of object classes in dataset, a scalar
        spatial_size: tuple containing (height, width) of full-size image, a tuple
        base_channels: number of channels in last generator & first discriminator layers, a scalar
        z_dim: number of dimensions in noise vector (z), a scalar
        n_upsample: number of downsampling (encoder) and upsampling (generator) operations, a scalar
        n_disc_layer:: number of discriminator layers, a scalar
        n_disc: number of discriminators (at different scales), a scalar
    '''
    def __init__(
        self,
        n_classes,
        spatial_size,
        base_channels=64,
        z_dim=256,
        n_upsample=6,
        n_disc_layers=3,
        n_disc=3,
    ):
        super().__init__()
        self.encoder = Encoder(
            spatial_size, z_dim=z_dim, n_downsample=n_upsample, base_channels=base_channels,
        )
        self.generator = Generator(
            n_classes, spatial_size, z_dim=z_dim, base_channels=base_channels, n_upsample=n_upsample,
        )
        self.discriminator = Discriminator(
            n_classes + 3, base_channels=base_channels, n_layers=n_disc_layers, n_discriminators=n_disc,
        )

    def forward(self, x, seg):
        ''' Performs a full forward pass for training. '''
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        x_fake = self.generate(z, seg)
        pred = self.discriminate(x_fake, seg)
        return x_fake, pred

    def encode(self, x):
        return self.encoder(x)

    def generate(self, z, seg):
        ''' Generates fake image from noise vector and segmentation. '''
        return self.generator(z, seg)

    def discriminate(self, x, seg):
        ''' Predicts whether input image is real. '''
        return self.discriminator(tf.concat((x, seg), dim=1))

    @staticmethod
    def sample_z(mu, logvar):
        ''' Samples noise vector with reparameterization trick. '''
        eps = tf.random.normal(mu.size(), device=mu.device).to(mu.dtype)
        return (logvar / 2).exp() * eps + mu

    @property
    def n_disc(self):
        return self.discriminator.n_discriminators
    
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
    Implements composite loss for GauGAN.
    GauGAN reuses the composite loss functions that Pix2PixHD does, except it replaces the LSGAN loss with [Hinge loss](https://paperswithcode.com/method/gan-hinge-loss). 
    It also imposes a soft (0.05 weight) Kullbach-Leibler divergence (KLD) loss term on the Gaussian statistics generated by the encoder.
    #### A debrief on KLD
    KLD measures how different two probability distributions are. In the case of N(miu, gamma^2 * I) learned by the encoder, KLD loss encourages the learned distribution to be close to a standard Gaussian. 
    For more information on implementation, check out Pytorch's [KLDivLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html) documentation.

    Values:
        lambda1: weight for feature matching loss, a float
        lambda2: weight for vgg perceptual loss, a float
        lambda3: weight for KLD loss, a float
        device: 'cuda' or 'cpu' for hardware to use
        norm_weight_to_one: whether to normalize weights to (0, 1], a bool
    '''
    def __init__(self, lambda1=10., lambda2=10., lambda3=0.05, device='cuda', norm_weight_to_one=True):
        super().__init__()

        self.vgg = _VGG19()
        self.vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

        lambda0 = 1.0
        # Keep ratio of composite loss, but scale down max to 1.0
        scale = max(lambda0, lambda1, lambda2, lambda3) if norm_weight_to_one else 1.0

        self.lambda0 = lambda0 / scale
        self.lambda1 = lambda1 / scale
        self.lambda2 = lambda2 / scale
        self.lambda3 = lambda3 / scale

    def kld_loss(self, mu, logvar):
        return -0.5 * tf.math.reduce_sum(1 + logvar - mu ** 2 - logvar.exp())

    def g_adv_loss(self, discriminator_preds):
        adv_loss = 0.0
        for preds in discriminator_preds:
            pred = preds[-1]
            adv_loss += -pred.mean()
        return adv_loss

    def d_adv_loss(self, discriminator_preds, is_real):
        adv_loss = 0.0
        for preds in discriminator_preds:
            pred = preds[-1]
            target = -1 + pred if is_real else -1 - pred
            mask = target < 0
            adv_loss += (mask * target).mean()
        return adv_loss

    def fm_loss(self, real_preds, fake_preds):
        fm_loss = 0.0
        for real_features, fake_features in zip(real_preds, fake_preds):
            for real_feature, fake_feature in zip(real_features, fake_features):
                fm_loss += MeanAbsoluteError(real_feature, fake_feature) # Original code has .detach() on the real_feature Pytorch tensor
        return fm_loss

    def vgg_loss(self, x_real, x_fake):
        vgg_real = self.vgg(x_real)
        vgg_fake = self.vgg(x_fake)

        vgg_loss = 0.0
        for real, fake, weight in zip(vgg_real, vgg_fake, self.vgg_weights):
            vgg_loss += weight * MeanAbsoluteError(real, fake) # Original code has .detach() on the real Pytorch tensor
        return vgg_loss

    def call(self, x_real, label_map, gaugan):
        '''
        Function that computes the forward pass and total loss for GauGAN.
        '''
        mu, logvar = gaugan.encode(x_real)
        z = gaugan.sample_z(mu, logvar)
        x_fake = gaugan.generate(z, label_map)

        # Get necessary outputs for loss/backprop for both generator and discriminator
        fake_preds_for_g = gaugan.discriminate(x_fake, label_map)
        fake_preds_for_d = gaugan.discriminate(x_fake, label_map) # Original code has .detach() on the x_fake Pytorch tensor
        real_preds_for_d = gaugan.discriminate(x_real, label_map) # Original code has .detach() on the x_real Pytorch tensor

        g_loss = (
            self.lambda0 * self.g_adv_loss(fake_preds_for_g) + \
            self.lambda1 * self.fm_loss(real_preds_for_d, fake_preds_for_g) / gaugan.n_disc + \
            self.lambda2 * self.vgg_loss(x_fake, x_real) + \
            self.lambda3 * self.kld_loss(mu, logvar)
        )
        d_loss = 0.5 * (
            self.d_adv_loss(real_preds_for_d, True) + \
            self.d_adv_loss(fake_preds_for_d, False)
        )
        return g_loss, d_loss, x_fake # Original code has .detach() on the x_fake Pytorch tensor

# https://www.tensorflow.org/datasets/catalog/cityscapes
# https://www.cityscapes-dataset.com/
epochs = 200                    # total number of train epochs
decay_after = 100               # number of epochs with constant lr
betas = (0.0, 0.999)
def lr_lambda(epoch):
    ''' Function for scheduling learning rate '''
    return 1. if epoch < decay_after else 1 - float(epoch - decay_after) / (epochs - decay_after)

gaugan_config = {
    'n_classes': 35,
    'spatial_size': (128, 256), # Default (256, 512): halve size for memory
    'base_channels': 32,        # Default 64: halve channels for memory
    'z_dim': 256,
    'n_upsample': 5,            # Default 6: decrease layers for memory
    'n_disc_layers': 2,
    'n_disc': 3,
}
gaugan = GauGAN(**gaugan_config)
loss = Loss()

# Initialize dataloader
train_dir = ['data']
batch_size = 16                 # Default 32: decrease for memory
#dataset = CityscapesDataset(
#    train_dir, img_size=gaugan_config['spatial_size'], n_classes=gaugan_config['n_classes'],
#)
# def __init__(self, paths, img_size=(256, 512), n_classes=35):
dataloader = GauGANCityscapeDataGenerator("data/cityscape/", img_size=gaugan_config['spatial_size'], n_classes=gaugan_config['n_classes'])
#dataloader = DataLoader(
#    dataset, collate_fn=CityscapesDataset.collate_fn,
#    batch_size=batch_size, shuffle=True,
#    drop_last=False, pin_memory=True,
#)

# Initialize optimizers + schedulers
epochs = 200                    # total number of train epochs
decay_after = 100               # number of epochs with constant lr
betas = (0.0, 0.999)

g_params = list(gaugan.generator.parameters()) + list(gaugan.encoder.parameters())
d_params = list(gaugan.discriminator.parameters())

g_optimizer = Adam(g_params, learning_rate=lr_lambda, beta_1 = betas[0], beta_2 = betas[1])
d_optimizer = Adam(d_params, learning_rate=lr_lambda, beta_1 = betas[0], beta_2 = betas[1])

def Train(dataloader, gaugan, optimizers):
    g_optimizer, d_optimizer = optimizers

    cur_step = 0
    display_step = 100

    mean_g_loss = 0.0
    mean_d_loss = 0.0

    for epoch in tqdm(range(epochs)):
        start = time.time()
        for (x_real, labels) in tqdm(dataloader, position=0):

            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                g_loss, d_loss, x_fake = loss(x_real, labels, gaugan)

            # Compute gradients
            g_gradients = gen_tape.gradient(g_loss, gaugan.generator.trainable_variables)
            d_gradients = disc_tape.gradient(d_loss, gaugan.discriminator.trainable_variables)

            # Apply gradients to update weights to model
            g_optimizer.apply_gradients(zip(g_gradients, gaugan.generator.trainable_variables))
            d_optimizer.apply_gradients(zip(d_gradients, gaugan.discriminator.trainable_variables))

            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'
                      .format(cur_step, mean_g_loss, mean_d_loss))
                show_tensor_images(x_fake.to(x_real.dtype))
                show_tensor_images(x_real)
                mean_g_loss = 0.0
                mean_d_loss = 0.0
            cur_step += 1
        print(f"Epoch {epoch + 1}: {time.time()-start}s")

Train(
    dataloader, gaugan,
    [g_optimizer, d_optimizer],
)