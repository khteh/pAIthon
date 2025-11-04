import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Normalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotModelHistory
from .SignsLanguageDigits import SignsLanguageDigits

class RNN_SignsLanguageDigits(SignsLanguageDigits):
    """
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
    Very deep "plain" networks don't work in practice because vanishing gradients make them hard to train.
    Skip connections help address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function.
    There are two main types of blocks: The identity block and the convolutional block.
    Very deep Residual Networks are built by stacking these blocks together.    
    """
    def BuildModel(self):
        """
        The details of this ResNet-50 model are:
        - Zero-padding pads the input with a pad of (3,3)
        - Stage 1:
            - The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). 
            - BatchNorm is applied to the 'channels' axis of the input.
            - MaxPooling uses a (3,3) window and a (2,2) stride.
        - Stage 2:
            - The convolutional block uses three sets of filters of size [64,64,256], "f" is 3, and "s" is 1.
            - The 2 identity blocks use three sets of filters of size [64,64,256], and "f" is 3.
        - Stage 3:
            - The convolutional block uses three sets of filters of size [128,128,512], "f" is 3 and "s" is 2.
            - The 3 identity blocks use three sets of filters of size [128,128,512] and "f" is 3.
        - Stage 4:
            - The convolutional block uses three sets of filters of size [256, 256, 1024], "f" is 3 and "s" is 2.
            - The 5 identity blocks use three sets of filters of size [256, 256, 1024] and "f" is 3.
        - Stage 5:
            - The convolutional block uses three sets of filters of size [512, 512, 2048], "f" is 3 and "s" is 2.
            - The 2 identity blocks use three sets of filters of size [512, 512, 2048] and "f" is 3.
        - The 2D Average Pooling uses a window (pool_size) of shape (2,2).
        - The 'flatten' layer doesn't have any hyperparameters.
        - The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation.

        Stage-wise implementation of the architecture of the popular ResNet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        if not self._model:
            # Define the input as a tensor with shape input_shape
            X_input = Input(self._input_shape)
            X_input = Normalization(axis=-1)(X_input)
            # Zero-Padding
            X = ZeroPadding2D((3, 3))(X_input)
            
            # Stage 1
            X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis = -1)(X)
            X = Activation('relu')(X)
            X = Dropout(0.3)(X)
            X = MaxPooling2D((3, 3), strides=(2, 2))(X)

            # Stage 2
            X = self._convolutional_block(X, f = 3, filters = [64, 64, 256], name="Conv_Stage2", s = 1)
            X = self._identity_block(X, 3, [64, 64, 256], "ID_Stage2.1")
            X = self._identity_block(X, 3, [64, 64, 256], "ID_Stage2.2")

            # Use the instructions above in order to implement all of the Stages below
            # Make sure you don't miss adding any required parameter
            
            ## Stage 3 (≈4 lines)
            # `_convolutional_block` with correct values of `f`, `filters` and `s` for this stage
            X = self._convolutional_block(X, f = 3, filters = [128, 128, 512], name="Conv_Stage3", s = 2)
            
            # the 3 `_identity_block` with correct values of `f` and `filters` for this stage
            X = self._identity_block(X, 3, [128, 128, 512], "ID_Stage3.1")
            X = self._identity_block(X, 3, [128, 128, 512], "ID_Stage3.2")
            X = self._identity_block(X, 3, [128, 128, 512], "ID_Stage3.3")
            
            # Stage 4 (≈6 lines)
            # add `_convolutional_block` with correct values of `f`, `filters` and `s` for this stage
            X = self._convolutional_block(X, f = 3, filters = [256, 256, 1024], name="Conv_Stage4", s = 2)
            
            # the 5 `_identity_block` with correct values of `f` and `filters` for this stage
            X = self._identity_block(X, 3, [256, 256, 1024], "ID_Stage4.1")
            X = self._identity_block(X, 3, [256, 256, 1024], "ID_Stage4.2")
            X = self._identity_block(X, 3, [256, 256, 1024], "ID_Stage4.3")
            X = self._identity_block(X, 3, [256, 256, 1024], "ID_Stage4.4")
            X = self._identity_block(X, 3, [256, 256, 1024], "ID_Stage4.5")

            # Stage 5 (≈3 lines)
            # add `_convolutional_block` with correct values of `f`, `filters` and `s` for this stage
            X = self._convolutional_block(X, f = 3, filters = [512, 512, 2048], name="Conv_Stage5", s = 2)
            
            # the 2 `_identity_block` with correct values of `f` and `filters` for this stage
            X = self._identity_block(X, 3, [512, 512, 2048], "ID_Stage5.1")
            X = self._identity_block(X, 3, [512, 512, 2048], "ID_Stage5.2")

            # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
            X = AveragePooling2D((2,2))(X)
            
            # output layer
            X = Flatten()(X)
            X = Dense(self._classes, kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=l2(0.01))(X) # Decrease to fix high bias; Increase to fix high variance.
            # Create model
            self._model = Model(inputs = X_input, outputs = X)
            self._model.name = self._name
            self._model.compile(
                    loss=CategoricalCrossentropy(from_logits=True), # Logistic Loss: -ylog(f(X)) - (1 - y)log(1 - f(X)) Defaults to softmax activation which is typically used for multiclass classification
                    optimizer=Adam(learning_rate=self._learning_rate), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
                    metrics=['accuracy']
                )
            self._model.summary()
            """
            https://github.com/keras-team/keras/issues/21815
            plot_model(
                self._model,
                to_file=f"output/{self._name}.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
            """
    def TrainModel(self, epochs:int, retrain: bool = False):
        super().TrainModel(epochs, False, retrain)

    def _identity_block(self, X, f, filters, name:str, initializer=random_uniform):
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
        #print(f"X: {X.shape}")
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0), name=f"{name}.Conv1")(X)
        X = BatchNormalization(axis = -1)(X) # Default axis # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        X = Activation('relu')(X)
        X = Dropout(0.3, name=f"{name}.Dropout1")(X)

        ## Second component of main path (≈3 lines)
        ## Set the padding = 'same'
        X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0), name=f"{name}.Conv2")(X)
        X = BatchNormalization(axis = -1)(X) # Default axis # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        X = Activation('relu')(X)
        X = Dropout(0.3, name=f"{name}.Dropout2")(X)

        ## Third component of main path (≈2 lines)
        ## Set the padding = 'valid'
        X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0), name=f"{name}.Conv3")(X)
        X = BatchNormalization(axis = -1)(X) # Default axis # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        
        ## Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        X = Dropout(0.3, name=f"{name}.Dropout3")(X)
        return X

    def _convolutional_block(self, X, f, filters, name:str, s = 2, initializer=glorot_uniform):
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
        X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0), name=f"{name}.Conv1")(X)
        X = BatchNormalization(axis = -1)(X) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        X = Activation('relu')(X)
        X = Dropout(0.3, name=f"{name}.Dropout1")(X)
        #print(f"X1: {X.shape}")
        
        ## Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer(seed=0), name=f"{name}.Conv2")(X)
        X = BatchNormalization(axis = -1)(X) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        X = Activation('relu')(X)
        X = Dropout(0.3, name=f"{name}.Dropout2")(X)
        #print(f"X2: {X.shape}")
        
        ## Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed=0), name=f"{name}.Conv3")(X)
        X = BatchNormalization(axis = -1)(X) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        #print(f"X3: {X.shape}")
        
        ##### SHORTCUT PATH ##### (≈2 lines)
        X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0), name=f"{name}.Conv4")(X_shortcut)
        X_shortcut = BatchNormalization(axis = -1)(X_shortcut) # stabilize the learning process, accelerate convergence (speed up training), and potentially improve generalization performance.
        #print(f"shortcut: {X_shortcut.shape}")

        # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        X = Dropout(0.3, name=f"{name}.Dropout3")(X)
        return X
    
if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='RNN Signs Language multi-class classifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    parser.add_argument('-g', '--grayscale', action='store_true', help='Use grayscale model')
    args = parser.parse_args()
    model = f"models/RNN_SignsLanguageDigits_{'grayscale' if args.grayscale else 'RGB'}.keras"
    signs = RNN_SignsLanguageDigits("RNN_SignsLanguageDigits", args.grayscale, model, (64, 64, 1 if args.grayscale else 3), 32, 0.0001)
    signs.BuildModel()
    #InitializeGPU()
    signs.TrainModel(400, args.retrain)
    signs.PredictSign("images/my_handsign0.jpg", 2, args.grayscale)
    signs.PredictSign("images/my_handsign1.jpg", 1, args.grayscale)
    signs.PredictSign("images/my_handsign2.jpg", 3, args.grayscale)
    signs.PredictSign("images/my_handsign3.jpg", 5, args.grayscale)
    signs.PredictSign("images/my_handsign4.jpg", 5, args.grayscale)