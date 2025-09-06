import math, numpy
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
import scipy.misc
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from resnets_utils import *
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
from utils.GPU import InitializeGPU

# convolutional_block_output1 = [[[[0.,         0.,         0.6442667,  0.,         0.13945118, 0.78498244],
#                                  [0.01695363, 0.,         0.7052939,  0.,         0.27986753, 0.67453355]],
#                                 [[0.,         0.,         0.6702033,  0. ,        0.18277727, 0.7506114 ],
#                                  [0.,         0.,         0.68768525, 0. ,        0.25753927, 0.6794529 ]]],
#                                [[[0.,         0.7772112,  0.,        1.4986887,  0.,         0.        ],
#                                  [0.,         1.0264266,  0.,        1.274425,   0.,         0.        ]],
#                                 [[0.,         1.0375856,  0.,        1.6640364,  0.,         0.        ],
#                                  [0.,         1.0398285,  0.,        1.3485202,  0.,         0.        ]]],
#                                [[[0.,         2.3315008,  0.,        4.4961185,  0.,         0.        ],
#                                  [0.,         3.0792732,  0.,        3.8233364,  0.,         0.        ]],
#                                 [[0.,         3.1125813,  0.,        4.9924607,  0.,         0.        ],
#                                  [0.,         3.1193442,  0.,        4.0456157,  0.,         0.        ]]]]

# convolutional_block_output2 = [[[[0.0000000e+00, 2.4476275e+00, 1.8830043e+00, 2.1259236e-01, 1.9220030e+00, 0.0000000e+00],
#                                  [0.0000000e+00, 2.1546977e+00, 1.6514317e+00, 0.0000000e+00, 1.7889941e+00, 0.0000000e+00]],
#                                 [[0.0000000e+00, 1.8540058e+00, 1.3404746e+00, 0.0000000e+00, 1.0688392e+00, 0.0000000e+00],
#                                  [0.0000000e+00, 1.6571904e+00, 1.1809819e+00, 0.0000000e+00, 9.4837922e-01, 0.0000000e+00]]],
#                                [[[0.0000000e+00, 5.0503787e-02, 0.0000000e+00, 2.9122047e-03, 8.7130928e-01, 1.0279868e+00],
#                                  [0.0000000e+00, 5.0503787e-02, 0.0000000e+00, 2.9122047e-03, 8.7130928e-01, 1.0279868e+00]],
#                                 [[0.0000000e+00, 5.0503787e-02, 0.0000000e+00, 2.9122047e-03, 8.7130928e-01, 1.0279868e+00],
#                                  [0.0000000e+00, 5.0503787e-02, 0.0000000e+00, 2.9122047e-03, 8.7130928e-01, 1.0279868e+00]]],
#                                [[[1.9959736e+00, 0.0000000e+00, 0.0000000e+00, 2.4793634e+00, 0.0000000e+00, 2.9498351e-01],
#                                  [1.4637939e+00, 0.0000000e+00, 0.0000000e+00, 1.3023224e+00, 0.0000000e+00, 1.5583299e+00]],
#                                 [[3.1462767e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.1307199e+00],
#                                  [1.8378723e+00, 0.0000000e+00, 0.0000000e+00, 1.5683722e-01, 0.0000000e+00, 2.3509054e+00]]]]

ResNet50_summary =[['InputLayer', [(None, 64, 64, 3)], 0],
['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
['Conv2D', (None, 32, 32, 64), 9472, 'valid', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 32, 32, 64), 256],
['Activation', (None, 32, 32, 64), 0],
['MaxPooling2D', (None, 15, 15, 64), 0, (3, 3), (2, 2), 'valid'],
['Conv2D', (None, 15, 15, 64), 4160, 'valid', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 15, 15, 64), 256],
['Activation', (None, 15, 15, 64), 0],
['Conv2D', (None, 15, 15, 64), 36928, 'same', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 15, 15, 64), 256],
['Activation', (None, 15, 15, 64), 0],
['Conv2D', (None, 15, 15, 256), 16640, 'valid', 'linear', 'GlorotUniform'],
['Conv2D', (None, 15, 15, 256), 16640, 'valid', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 15, 15, 256), 1024],
['BatchNormalization', (None, 15, 15, 256), 1024],
['Add', (None, 15, 15, 256), 0],
['Activation', (None, 15, 15, 256), 0],
['Conv2D', (None, 15, 15, 64), 16448, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 15, 15, 64), 256],
['Activation', (None, 15, 15, 64), 0],
['Conv2D', (None, 15, 15, 64), 36928, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 15, 15, 64), 256],
['Activation', (None, 15, 15, 64), 0],
['Conv2D', (None, 15, 15, 256), 16640, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 15, 15, 256), 1024],
['Add', (None, 15, 15, 256), 0],
['Activation', (None, 15, 15, 256), 0],
['Conv2D', (None, 15, 15, 64), 16448, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 15, 15, 64), 256],
['Activation', (None, 15, 15, 64), 0],
['Conv2D', (None, 15, 15, 64), 36928, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 15, 15, 64), 256],
['Activation', (None, 15, 15, 64), 0],
['Conv2D', (None, 15, 15, 256), 16640, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 15, 15, 256), 1024],
['Add', (None, 15, 15, 256), 0],
['Activation', (None, 15, 15, 256), 0],
['Conv2D', (None, 8, 8, 128), 32896, 'valid', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 8, 8, 128), 512],
['Activation', (None, 8, 8, 128), 0],
['Conv2D', (None, 8, 8, 128), 147584, 'same', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 8, 8, 128), 512],
['Activation', (None, 8, 8, 128), 0],
['Conv2D', (None, 8, 8, 512), 66048, 'valid', 'linear', 'GlorotUniform'],
['Conv2D', (None, 8, 8, 512), 131584, 'valid', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 8, 8, 512), 2048],
['BatchNormalization', (None, 8, 8, 512), 2048],
['Add', (None, 8, 8, 512), 0],
['Activation', (None, 8, 8, 512), 0],
['Conv2D', (None, 8, 8, 128), 65664, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 8, 8, 128), 512],
['Activation', (None, 8, 8, 128), 0],
['Conv2D', (None, 8, 8, 128), 147584, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 8, 8, 128), 512],
['Activation', (None, 8, 8, 128), 0],
['Conv2D', (None, 8, 8, 512), 66048, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 8, 8, 512), 2048],
['Add', (None, 8, 8, 512), 0],
['Activation', (None, 8, 8, 512), 0],
['Conv2D', (None, 8, 8, 128), 65664, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 8, 8, 128), 512],
['Activation', (None, 8, 8, 128), 0],
['Conv2D', (None, 8, 8, 128), 147584, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 8, 8, 128), 512],
['Activation', (None, 8, 8, 128), 0],
['Conv2D', (None, 8, 8, 512), 66048, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 8, 8, 512), 2048],
['Add', (None, 8, 8, 512), 0],
['Activation', (None, 8, 8, 512), 0],
['Conv2D', (None, 8, 8, 128), 65664, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 8, 8, 128), 512],
['Activation', (None, 8, 8, 128), 0],
['Conv2D', (None, 8, 8, 128), 147584, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 8, 8, 128), 512],
['Activation', (None, 8, 8, 128), 0],
['Conv2D', (None, 8, 8, 512), 66048, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 8, 8, 512), 2048],
['Add', (None, 8, 8, 512), 0],
['Activation', (None, 8, 8, 512), 0],
['Conv2D', (None, 4, 4, 256), 131328, 'valid', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 256), 590080, 'same', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 1024), 263168, 'valid', 'linear', 'GlorotUniform'],
['Conv2D', (None, 4, 4, 1024), 525312, 'valid', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 4, 4, 1024), 4096],
['BatchNormalization', (None, 4, 4, 1024), 4096],
['Add', (None, 4, 4, 1024), 0],
['Activation', (None, 4, 4, 1024), 0],
['Conv2D', (None, 4, 4, 256), 262400, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 256), 590080, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 1024), 263168, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 1024), 4096],
['Add', (None, 4, 4, 1024), 0],
['Activation', (None, 4, 4, 1024), 0],
['Conv2D', (None, 4, 4, 256), 262400, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 256), 590080, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 1024), 263168, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 1024), 4096],
['Add', (None, 4, 4, 1024), 0],
['Activation', (None, 4, 4, 1024), 0],
['Conv2D', (None, 4, 4, 256), 262400, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 256), 590080, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 1024), 263168, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 1024), 4096],
['Add', (None, 4, 4, 1024), 0],
['Activation', (None, 4, 4, 1024), 0],
['Conv2D', (None, 4, 4, 256), 262400, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 256), 590080, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 1024), 263168, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 1024), 4096],
['Add', (None, 4, 4, 1024), 0],
['Activation', (None, 4, 4, 1024), 0],
['Conv2D', (None, 4, 4, 256), 262400, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 256), 590080, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 256), 1024],
['Activation', (None, 4, 4, 256), 0],
['Conv2D', (None, 4, 4, 1024), 263168, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 4, 4, 1024), 4096],
['Add', (None, 4, 4, 1024), 0],
['Activation', (None, 4, 4, 1024), 0],
['Conv2D', (None, 2, 2, 512), 524800, 'valid', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 2, 2, 512), 2048],
['Activation', (None, 2, 2, 512), 0],
['Conv2D', (None, 2, 2, 512), 2359808, 'same', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 2, 2, 512), 2048],
['Activation', (None, 2, 2, 512), 0],
['Conv2D', (None, 2, 2, 2048), 1050624, 'valid', 'linear', 'GlorotUniform'],
['Conv2D', (None, 2, 2, 2048), 2099200, 'valid', 'linear', 'GlorotUniform'],
['BatchNormalization', (None, 2, 2, 2048), 8192],
['BatchNormalization', (None, 2, 2, 2048), 8192],
['Add', (None, 2, 2, 2048), 0],
['Activation', (None, 2, 2, 2048), 0],
['Conv2D', (None, 2, 2, 512), 1049088, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 2, 2, 512), 2048],
['Activation', (None, 2, 2, 512), 0],
['Conv2D', (None, 2, 2, 512), 2359808, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 2, 2, 512), 2048],
['Activation', (None, 2, 2, 512), 0],
['Conv2D', (None, 2, 2, 2048), 1050624, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 2, 2, 2048), 8192],
['Add', (None, 2, 2, 2048), 0],
['Activation', (None, 2, 2, 2048), 0],
['Conv2D', (None, 2, 2, 512), 1049088, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 2, 2, 512), 2048],
['Activation', (None, 2, 2, 512), 0],
['Conv2D', (None, 2, 2, 512), 2359808, 'same', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 2, 2, 512), 2048],
['Activation', (None, 2, 2, 512), 0],
['Conv2D', (None, 2, 2, 2048), 1050624, 'valid', 'linear', 'RandomUniform'],
['BatchNormalization', (None, 2, 2, 2048), 8192],
['Add', (None, 2, 2, 2048), 0],
['Activation', (None, 2, 2, 2048), 0],
['AveragePooling2D', (None, 1, 1, 2048), 0],
['Flatten', (None, 2048), 0],
['Dense', (None, 6), 12294, 'softmax']]

class ResidualNetwork50():
    _model: Model = None
    _input_shape = None
    _classes: int = None
    _learning_rate: float = None
    """
    The details of this ResNet-50 model are:

    Zero-padding pads the input with a pad of (3,3)
    Stage 1:
    The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2).
    BatchNorm is applied to the 'channels' axis of the input.
    MaxPooling uses a (3,3) window and a (2,2) stride.
    Stage 2:
    The convolutional block uses three sets of filters of size [64,64,256], "f" is 3, and "s" is 1.
    The 2 identity blocks use three sets of filters of size [64,64,256], and "f" is 3.
    Stage 3:
    The convolutional block uses three sets of filters of size [128,128,512], "f" is 3 and "s" is 2.
    The 3 identity blocks use three sets of filters of size [128,128,512] and "f" is 3.
    Stage 4:
    The convolutional block uses three sets of filters of size [256, 256, 1024], "f" is 3 and "s" is 2.
    The 5 identity blocks use three sets of filters of size [256, 256, 1024] and "f" is 3.
    Stage 5:
    The convolutional block uses three sets of filters of size [512, 512, 2048], "f" is 3 and "s" is 2.
    The 2 identity blocks use three sets of filters of size [512, 512, 2048] and "f" is 3.
    The 2D Average Pooling uses a window (pool_size) of shape (2,2).
    The 'flatten' layer doesn't have any hyperparameters.
    The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation.    
    """
    def __init__(self, input_shape, classes:int, learning_rate:float):
        InitializeGPU()
        self._input_shape = input_shape
        self._classes = classes
        self._learning_rate = learning_rate

    def identity_block(self, X, f, filters, initializer=random_uniform):
        """
        Implementation of the identity block as defined in Figure 4
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
        
        Returns:
        X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
        """
        print(f"X: {X.shape}")
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
        X = layers.BatchNormalization(axis = 3)(X) # Default axis # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        X = Activation('relu')(X)
        
        ### START CODE HERE
        ## Second component of main path (≈3 lines)
        ## Set the padding = 'same'
        X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
        X = layers.BatchNormalization(axis = 3)(X) # Default axis # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        X = Activation('relu')(X)

        ## Third component of main path (≈2 lines)
        ## Set the padding = 'valid'
        X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
        X = layers.BatchNormalization(axis = 3)(X) # Default axis # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        
        ## Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        ### END CODE HERE

        return X

    def convolutional_block(self, X, f, filters, s = 2, initializer=glorot_uniform):
        """
        Implementation of the convolutional block as defined in Figure 4
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        s -- Integer, specifying the stride to be used
        initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                    also called Xavier uniform initializer.
        
        Returns:
        X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
        """
        #print(f"X: {X.shape}, Filters: {filters}, f: {f}, strides: {s}")
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        
        # First component of main path glorot_uniform(seed=0)
        X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
        X = layers.BatchNormalization(axis = 3)(X) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        X = Activation('relu')(X)
        #print(f"X1: {X.shape}")
        ### START CODE HERE
        
        ## Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer(seed=0))(X)
        X = layers.BatchNormalization(axis = 3)(X) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        X = Activation('relu')(X)
        #print(f"X2: {X.shape}")
        
        ## Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed=0))(X)
        X = layers.BatchNormalization(axis = 3)(X) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        #print(f"X3: {X.shape}")
        
        ##### SHORTCUT PATH ##### (≈2 lines)
        X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X_shortcut)
        X_shortcut = layers.BatchNormalization(axis = 3)(X_shortcut) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        #print(f"shortcut: {X_shortcut.shape}")
        ### END CODE HERE

        # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        
        return X
    
    def BuildModel(self):
        """
        Stage-wise implementation of the architecture of the popular ResNet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        tf.keras.backend.set_learning_phase(True)
        # Define the input as a tensor with shape input_shape
        X_input = Input(self._input_shape)
        
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        
        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
        X = layers.BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
        X = self.identity_block(X, 3, [64, 64, 256])
        X = self.identity_block(X, 3, [64, 64, 256])

        ### START CODE HERE
        
        # Use the instructions above in order to implement all of the Stages below
        # Make sure you don't miss adding any required parameter
        
        ## Stage 3 (≈4 lines)
        # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
        X = self.convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
        
        # the 3 `identity_block` with correct values of `f` and `filters` for this stage
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])
        
        # Stage 4 (≈6 lines)
        # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
        X = self.convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
        
        # the 5 `identity_block` with correct values of `f` and `filters` for this stage
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])

        # Stage 5 (≈3 lines)
        # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
        X = self.convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
        
        # the 2 `identity_block` with correct values of `f` and `filters` for this stage
        X = self.identity_block(X, 3, [512, 512, 2048])
        X = self.identity_block(X, 3, [512, 512, 2048])

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
        X = AveragePooling2D()(X)
        
        ### END CODE HERE

        # output layer
        X = Flatten()(X)
        X = Dense(self._classes, kernel_initializer = glorot_uniform(seed=0))(X)
        # Create model
        self._model = Model(inputs = X_input, outputs = X)
        self._model.compile(
                loss=CategoricalCrossentropy(from_logits=True), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
                optimizer=Adam(learning_rate=self._learning_rate), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                metrics=['accuracy']
            )
        self._model.summary()
    
if __name__ == "__main__":
    resnet50 = ResidualNetwork50((64, 64, 3), 6, 0.00015)