import matplotlib.pyplot as plt, pandas as pd, numpy, cv2, shap, tensorflow as tf
import tensorflow.keras.backend as K
from keras.preprocessing import image
from tensorflow.keras import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

class GradCAM():
    """
    Grad-CAM, a powerful technique for interpreting Convolutional Neural Networks. Grad-CAM stands for Gradient-weighted Class Activation Mapping.
    CNN's are very flexible models and their great predictive power comes at the cost of losing interpretability (something that is true for all Artificial Neural Networks). 
    Grad-CAM attempts to solve this by giving us a graphical visualisation of parts of an image that are the most relevant for the CNN when predicting a particular class.
    """
    _path:str = None
    _labels = None
    _train_df = None
    _valid_df = None
    _test_df = None
    _pos_weights = None
    _neg_weights = None
    _mean = None
    _std = None
    def __init__(self, path:str):
        self._path = path
        self._PrepareData()

    def BuildModel(self):
        # create the base pre-trained model
        base_model = DenseNet121(weights='models/densenet.hdf5', include_top=False)
        print("Loaded DenseNet")
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # and a logistic layer
        predictions = Dense(len(self._labels), activation="sigmoid")(x)
        print("Added layers")

        self._model = Model(inputs=base_model.input, outputs=predictions)

        def get_weighted_loss(neg_weights, pos_weights, epsilon=1e-7):
            def weighted_loss(y_true, y_pred):
                # L(X, y) = −w * y log p(Y = 1|X) − w *  (1 − y) log p(Y = 0|X)
                # from https://arxiv.org/pdf/1711.05225.pdf
                loss = 0
                for i in range(len(neg_weights)):
                    loss -= (neg_weights[i] * y_true[:, i] * tf.math.log(y_pred[:, i] + epsilon) + 
                            pos_weights[i] * (1 - y_true[:, i]) * tf.math.log(1 - y_pred[:, i] + epsilon))
                loss = tf.math.reduce_sum(loss)
                return loss
            return weighted_loss
        self._model.compile(
            loss=get_weighted_loss(self._neg_weights, self._pos_weights),
            optimizer=Adam(), # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
        )
        self._model.load_weights("models/pretrained_model.h5")
        self._model.summary()

    def compute_gradcam(self, img, selected_labels):
        """
        Compute GradCAM for many specified labels for an image. 
        This method will use the `grad_cam` function.
        
        Args:
            model (Keras.model): Model to compute GradCAM for
            img (string): Image name we want to compute GradCAM for.
            mean (float): Mean to normalize to image.
            std (float): Standard deviation to normalize the image.
            data_dir (str): Path of the directory to load the images from.
            df(pd.Dataframe): Dataframe with the image features.
            labels ([str]): All output labels for the model.
            selected_labels ([str]): All output labels we want to compute the GradCAM for.
            layer_name: Intermediate layer from the model we want to compute the GradCAM for.
        """
        print(f"\n=== {self.compute_gradcam.__name__} ===")
        img_path = f"{self._path}/images-small/{img}"
        preprocessed_input = self._load_image_normalize(img_path, self._mean, self._std)
        layer_name='conv5_block16_concat'
        with tf.GradientTape() as tape:
            predictions = self._model.predict(preprocessed_input)
            spatial_map_layer = self._model.get_layer(layer_name).output
            # Watch the intermediate layer's output
            tape.watch(spatial_map_layer) # https://github.com/tensorflow/tensorflow/issues/104521

        print("Ground Truth: ", ", ".join(numpy.take(self._labels, numpy.nonzero(self._train_df[self._train_df["Image"] == img][self._labels].values[0]))[0]))

        plt.figure(figsize=(15, 10))
        plt.subplot(151)
        plt.title("Original")
        plt.axis('off')
        plt.imshow(self._load_image(img_path, self._train_df, preprocess=False), cmap='gray')
        j = 1
        # Loop through all labels
        for i in range(len(self._labels)): # complete this line
            # Compute CAM and show plots for each selected label.
            
            # Check if the label is one of the selected labels
            if self._labels[i] in selected_labels: # complete this line
                
                # Use the grad_cam function to calculate gradcam
                # def grad_cam(input_model, image, category_index, layer_name):
                gradcam = self._grad_cam(preprocessed_input, i, tape)
                
                print("Generating gradcam for class %s (p=%2.2f)" % (self._labels[i], round(predictions[0][i], 3)))
                plt.subplot(151 + j)
                plt.title(self._labels[i] + ": " + str(round(predictions[0][i], 3)))
                plt.axis('off')
                plt.imshow(self._load_image(img_path, self._train_df, preprocess=False), cmap='gray')
                plt.imshow(gradcam, cmap='magma', alpha=min(0.5, predictions[0][i]))
                j +=1

    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        self._labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
        self._train_df = pd.read_csv(f"{self._path}/train-small.csv")
        self._valid_df = pd.read_csv(f"{self._path}/valid-small.csv")
        self._test_df = pd.read_csv(f"{self._path}/test.csv")
        self._get_mean_std_per_batch(f"{self._path}/images-small/00025288_001.png", self._train_df)

        class_pos = self._train_df.loc[:, self._labels].sum(axis=0)
        class_neg = len(self._train_df) - class_pos
        class_total = class_pos + class_neg

        self._pos_weights = class_pos / class_total
        self._neg_weights = class_neg / class_total

    def _grad_cam(self, image, category_index, tape):
        """
        GradCAM method for visualizing input saliency.
        
        Args:
            self._model (Keras.model): model to compute cam for
            image (tensor): input to model, shape (1, H, W, 3), where H (int) is height W (int) is width
            category_index (int): class to compute cam with respect to
            layer_name (str): relevant layer in model
        Return:
            cam ()
        """
        layer_name='conv5_block16_concat'
        cam = None
        
        # 1. Get placeholders for class output and last layer
        # Get the model's output
        output_with_batch_dim = self._model.output
        
        # Remove the batch dimension
        output_all_categories = output_with_batch_dim[0]
        
        # Retrieve only the disease category at the given category index
        y_c = output_all_categories[category_index]
        
        # Get the input model's layer specified by layer_name, and retrive the layer's output tensor
        spatial_map_layer = self._model.get_layer(layer_name).output

        # 2. Get gradients of last layer with respect to output

        # get the gradients of y_c with respect to the spatial map layer (it's a list of length 1)
        #grads_l = K.gradients(y_c, spatial_map_layer)
        gradients = tape.gradient(y_c, spatial_map_layer)
        
        # Get the gradient at index 0 of the list
        grads = gradients[0]
            
        # 3. Get hook for the selected layer and its gradient, based on given model's input
        # Hint: Use the variables produced by the previous two lines of code
        spatial_map_and_gradient_function = K.function([self._model.input], [spatial_map_layer, grads])
        
        # Put in the image to calculate the values of the spatial_maps (selected layer) and values of the gradients
        spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([image])

        # Reshape activations and gradient to remove the batch dimension
        # Shape goes from (B, H, W, C) to (H, W, C)
        # B: Batch. H: Height. W: Width. C: Channel    
        # Reshape spatial map output to remove the batch dimension
        spatial_map_val = spatial_map_all_dims[0]
        
        # Reshape gradients to remove the batch dimension
        grads_val = grads_val_all_dims[0]
        
        # 4. Compute weights using global average pooling on gradient 
        # grads_val has shape (Height, Width, Channels) (H,W,C)
        # Take the mean across the height and also width, for each channel
        # Make sure weights have shape (C)
        weights = numpy.mean(grads_val, axis=(0,1))
        print(f"weights: {weights.shape}")
        # 5. Compute dot product of spatial map values with the weights
        cam = spatial_map_val @ weights
        print(f"cam: {type(cam)}, {cam.shape}")
        
        # We'll take care of the postprocessing.
        H, W = image.shape[1], image.shape[2]
        cam = numpy.maximum(cam, 0) # ReLU so we only get positive importance
        cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
        cam = cam / cam.max()
        print(f"cam: {type(cam)}, {cam.shape}")
        return cam

    def _load_image_normalize(self, path, mean, std, H=320, W=320):
        x = image.load_img(path, target_size=(H, W))
        x -= mean
        x /= std
        return numpy.expand_dims(x, axis=0)

    def _load_image(self, path, df, preprocess=True, H = 320, W = 320):
        """Load and preprocess image."""
        x = image.load_img(path, target_size=(H, W))
        if preprocess:
            mean, std = self._get_mean_std_per_batch(df, H=H, W=W)
            x -= mean
            x /= std
            x = numpy.expand_dims(x, axis=0)
        return x
    
    def _get_mean_std_per_batch(self, image_path, df, H=320, W=320):
        sample_data = []
        for idx, img in enumerate(df.sample(100)["Image"].values):
            sample_data.append(numpy.array(image.load_img(image_path, target_size=(H, W))))
        self._mean = numpy.mean(sample_data[0])
        self._std = numpy.std(sample_data[0])

if __name__ == "__main__":
    gradcam = GradCAM("data/nih")
    gradcam.BuildModel()
    gradcam.compute_gradcam("00016650_000.png", ['Cardiomegaly', 'Mass', 'Edema'])
    gradcam.compute_gradcam("00005410_000.png", ['Cardiomegaly', 'Mass', 'Edema'])
    gradcam.compute_gradcam("00004090_002.png", ['Cardiomegaly', 'Mass', 'Edema'])