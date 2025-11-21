import matplotlib.pyplot as plt, pandas as pd, numpy, cv2, shap, tensorflow as tf
from pathlib import Path
from keras.preprocessing import image
from tensorflow.keras import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from utils.TermColour import bcolors
class GradCAM():
    """
    Grad-CAM, a powerful technique for interpreting Convolutional Neural Networks. Grad-CAM stands for Gradient-weighted Class Activation Mapping.
    CNN's are very flexible models and their great predictive power comes at the cost of losing interpretability (something that is true for all Artificial Neural Networks). 
    Grad-CAM attempts to solve this by giving us a graphical visualisation of parts of an image that are the most relevant for the CNN when predicting a particular class.
    """
    _path:str = None
    _labels = None
    _width:int = None
    _height:int = None
    _train_df = None
    _valid_df = None
    _test_df = None
    _pos_weights = None
    _neg_weights = None
    _mean = None
    _std = None
    _conv_layer = None
    _model: Model = None
    _grad_model: Model = None
    def __init__(self, path:str, width:int, height:int):
        self._path = path
        self._width = width
        self._height = height
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

        self._conv_layer = base_model.get_layer('conv5_block16_concat')
        self._model = Model(inputs=base_model.input, outputs=[self._conv_layer.output, predictions])
        self._grad_model = Model(
                                inputs=self._model.input,
                                outputs=[
                                    self._conv_layer.output,
                                    self._model.output
                                ]
                            )
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
        #self._model.summary()

    def compute_gradcam(self, img, selected_labels):
        """
        Compute GradCAM for many specified labels for an image. 
        This method will use the `grad_cam` function.
        https://keras.io/examples/vision/grad_cam/

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
        image = self._load_image(img_path, preprocess=False)
        image_normalized = self._load_image(img_path, preprocess=True)
        truth = numpy.nonzero(self._train_df[self._train_df["Image"] == img][self._labels].values[0])[0][0]
        truth_str = numpy.take(self._labels, numpy.nonzero(self._train_df[self._train_df["Image"] == img][self._labels].values[0]))[0][0]
        print(f"Image: {img} Ground Truth: {truth}, {truth_str}")
        fig, ax = plt.subplots(1, 4, constrained_layout=True, figsize=(15, 10)) # figsize = (width, height)
        # rect=[0, 0, 1, 0.98] tells tight_layout to arrange the subplots within the bottom 98% of the figure's height, leaving the top 2% some space for the suptitle, for instance.
        fig.tight_layout(rect=[0, 0, 1, 0.98]) #[left, bottom, right, top] Decrease the top boundary if the suptitle overlaps with the plots
        ax[0].set_title('Original', fontsize=22)
        ax[0].imshow(image, cmap='gray')
        ax[0].axis("off")
        j = 1
        # Loop through all labels
        for category in range(len(self._labels)): # complete this line
            # Compute CAM and show plots for each selected label.
            # Check if the label is one of the selected labels
            if self._labels[category] in selected_labels: # complete this line
                with tf.GradientTape() as tape:
                    """
                    https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call
                    This means that predict() calls can scale to very large arrays. Meanwhile, model(x) happens in-memory and doesn't scale. On the other hand, predict() is not differentiable: you cannot retrieve its gradient if you call it in a GradientTape scope.
                    You should use model(x) when you need to retrieve the gradients of the model call, and you should use predict() if you just need the output value. In other words, always use predict() unless you're in the middle of writing a low-level gradient descent loop (as we are now).
                    """
                    conv_outputs, predictions = self._model(image_normalized)

                    # Remove the batch dimension
                    # Retrieve only the disease category at the given category index
                    y_c = predictions[0][category]

                    # 2. Get gradients of last layer with respect to output
                    #print(f"Predictions: {predictions.shape} {predictions}, conv_output: {conv_outputs.shape} {conv_outputs}, y_c: {y_c}")

                # get the gradients of y_c with respect to the spatial map layer (it's a list of length 1)
                gradients = tape.gradient(y_c, conv_outputs)
                #print(f"gradients: {gradients}")
                # Get the gradient at index 0 of the list

                # Use the grad_cam function to calculate gradcam
                # def grad_cam(input_model, image, category_index, layer_name):
                # This is a vector where each entry is the mean intensity of the gradient
                # over a specific feature map channel
                pooled_grads = tf.reduce_mean(gradients, axis=(0, 1, 2))

                # We multiply each channel in the feature map array
                # by "how important this channel is" with regard to the top predicted class
                # then sum all the channels to obtain the heatmap class activation
                last_conv_layer_output = conv_outputs[0]
                heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis] # heatmap: EagerTensor
                heatmap = tf.squeeze(heatmap)

                # For visualization purpose, we will also normalize the heatmap between 0 & 1
                H, W = image_normalized.shape[1], image_normalized.shape[2]
                heatmap = tf.maximum(heatmap, 0) # ReLU so we only get positive importance
                #print(f"heatmap: {type(heatmap)}")
                heatmap = cv2.resize(heatmap.numpy(), (W, H), cv2.INTER_NEAREST)
                heatmap /= heatmap.max()

                pred_index = tf.argmax(predictions[0])
                colour = "black"
                termcolour = bcolors.DEFAULT
                if self._labels[category] == self._labels[pred_index]:
                    if pred_index == truth:
                        colour = "green"
                        termcolour = bcolors.OKGREEN
                    else:
                        colour = "red"
                        termcolour = bcolors.FAIL
                else:
                    #colour = "red"
                    termcolour = bcolors.FAIL
                    #print(f"{bcolors.FAIL} category: {category} {self._labels[category]}, prediction: {pred_index} {self._labels[pred_index]} {predictions[0]} {bcolors.DEFAULT}")
                print(f"{termcolour}Generating heatmap for class {self._labels[category]}, prediction: {self._labels[pred_index]} (p={predictions[0][category]:.4f}){bcolors.DEFAULT}")
                #print(f"truth: {truth}, pred_index: {pred_index}")
                ax[j].set_title(f"{self._labels[category]} ({predictions[0][category]:.4f})", fontsize=22, color=colour)
                ax[j].axis('off')
                ax[j].imshow(image, cmap='gray')
                ax[j].imshow(heatmap, cmap='magma', alpha=min(0.5, predictions[0][category].numpy()))
                j += 1
        fig.suptitle(f"{img} ({truth_str})", y=0.9, fontsize=22, fontweight="bold")
        plt.savefig(f"output/{Path(img).stem}_heatmap.png")

    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        self._labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
        print(f"{len(self._labels)} labels: {self._labels}")
        self._train_df = pd.read_csv(f"{self._path}/train-small.csv")
        self._valid_df = pd.read_csv(f"{self._path}/valid-small.csv")
        self._test_df = pd.read_csv(f"{self._path}/test.csv")
        self._get_mean_std_per_batch()

        class_pos = self._train_df.loc[:, self._labels].sum(axis=0)
        class_neg = len(self._train_df) - class_pos
        class_total = class_pos + class_neg

        self._pos_weights = class_pos / class_total
        self._neg_weights = class_neg / class_total

    def _load_image(self, path, preprocess=True):
        """Load and preprocess image."""
        x = image.load_img(path, target_size=(self._height, self._width))
        if preprocess:
            x -= self._mean
            x /= self._std
            x = numpy.expand_dims(x, axis=0)
        return x
    
    def _get_mean_std_per_batch(self):
        sample_data = []
        for idx, img in enumerate(self._train_df.sample(100)["Image"].values):
            sample_data.append(numpy.array(image.load_img(f"{self._path}/images-small/{img}", target_size=(self._height, self._width))))
        self._mean = numpy.mean(sample_data[0])
        self._std = numpy.std(sample_data[0])

if __name__ == "__main__":
    gradcam = GradCAM("data/nih", 320, 320)
    gradcam.BuildModel()
    gradcam.compute_gradcam("00016650_000.png", ['Cardiomegaly', 'Mass', 'Edema'])
    gradcam.compute_gradcam("00005410_000.png", ['Cardiomegaly', 'Mass', 'Edema'])
    gradcam.compute_gradcam("00004090_002.png", ['Cardiomegaly', 'Mass', 'Edema'])