import os, numpy, pandas as pd, tensorflow as tf, PIL
from tensorflow.keras.utils import plot_model, load_img
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from utils.TermColour import bcolors
K.set_image_data_format('channels_last')

class FaceRecognition():
    _model: Model = None
    _threshold: float = None
    _circuit_breaker = None
    _database = {}
    _names = None
    def __init__(self, threshold:bool):
        self._threshold = threshold
        self._BuildModel()
        self._PrepareData()

    def _BuildModel(self):
        """
        This network uses 299x299 dimensional RGB images as its input. Specifically, a face image (or batch of 𝑚 face images) as a tensor of shape  (𝑚,𝑛𝐻,𝑛𝑊,𝑛𝐶)=(𝑚,299,299,3)
        The input images are originally of shape 96x96, thus, you need to scale them to 299x299. This is done in the self._img_to_encoding() function.
        The output is a matrix of shape  (𝑚,128) that encodes each input face image into a 128-dimensional vector.
        By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. You then use the encodings to compare two face images as follows:
        https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2 width and height should be no smaller than 75. E.g. (150, 150, 3) would be one valid value.
        classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
        
        Default Behavior:
        By default, tf.keras.applications.InceptionResNetV2() is instantiated with include_top=True and pre-trained weights='imagenet'. In this configuration, the predict() method returns a vector of 1000 probabilities corresponding to the ImageNet object categories, not a general-purpose encoding vector. 
        Feature Extraction (Encoding Vector)
        To obtain a feature or encoding vector for use in transfer learning or other tasks, you should instantiate the model with specific parameters:
        include_top=False: This excludes the final fully-connected classification layer.
        pooling='avg' or pooling='max': This applies global average pooling or global max pooling to the output of the last convolutional block, converting the 4D tensor output into a 2D tensor (a vector for each image in the batch).         
        """
        if not self._model:
            self._model = InceptionResNetV2(include_top=False, pooling="avg")
            plot_model(
                self._model,
                to_file=f"output/FaceRecognition.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
        print(f"Model input: {self._model.inputs}")
        print(f"Model output: {self._model.outputs}")

    def _PrepareData(self):
        print(f"\n=== {self._PrepareData.__name__} ===")
        self._names = ["andrew", "arnaud", "benoit",  "bertrand", "dan", "danielle", "felix", "kevin", "kian", "sebastiano", "tian", "younes"]
        for n in self._names:
            path = f"images/{n}.png" if n == "danielle" else f"images/{n}.jpg"
            self._database[n] = self._img_to_encoding(path)

    def Verify(self, image_path:str, identity:str):
        """
        Face verification supervised learning:
        - 1:1 mapping
        - A 2-step authentication process:
            (i) Identiy who you are. For example, name, ID, etc.
            (ii) Check if the face matches the identity in step (i)
        Function that verifies if the person on the "image_path" image is "identity".
        
        Arguments:
            image_path -- path to an image
            identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office, i.e., a key in the database dictionary.
        
        Returns:
            dist -- distance between the image_path and the image of "identity" in the database.
            door_open -- True, if the door should open. False otherwise.
        """
        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        encoding = self._img_to_encoding(image_path)
        # Step 2: Compute distance with identity's image (≈ 1 line)
        # sqrt(sum((A - B) ** 2))
        dist = numpy.linalg.norm(encoding - self._database[identity], ord=2)
        print(f"dist: {dist}")
        # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
        if dist < self._threshold:
            print(f"{bcolors.OKGREEN}It's {identity}, welcome in!{bcolors.DEFAULT}")
            door_open = True
        else:
            print(f"{bcolors.FAIL}It's not {identity}, please leave!{bcolors.DEFAULT}")
            door_open = False
        return dist, door_open

    def Who_Is_It(self, image_path:str):
        """
        Implements face recognition for the office by finding who is the person on the image_path image.
        
        Arguments:
            image_path -- path to an image
            database -- database containing image encodings along with the name of the person on the image
            model -- your Inception model instance in Keras
        
        Returns:
            min_dist -- the minimum distance between image_path encoding and the encodings from the database
            identity -- string, the name prediction for the person on image_path
        """
        ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
        encoding =  self._img_to_encoding(image_path)
        
        ## Step 2: Find the closest encoding ##
        # Initialize "min_dist" to a large value, say 100 (≈1 line)
        min_dist = 100
        
        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in self._database.items():
            # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
            dist = numpy.linalg.norm(encoding - self._database[name], ord=2)

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist > self._threshold:
            print(f"{bcolors.FAIL}Not in the database.{bcolors.DEFAULT}")
        else:
            print (f"It's {identity}, the distance is {min_dist}")
        return min_dist, identity
    
    def _triplet_loss(self, y_pred, alpha = 0.2):
        """
        Implementation of the triplet loss as defined by formula (3)
        d(A,P) - d(A,N) + alpha <= 0
        L(A,P,N) = max(sum((f(A) - f(P))^2) - sum((f(A) - f(N))^2) + alpha, 0); sum over the image instances. 
        J = sum(L(A,P,N))for all the triplets over the image corpera and #persons. Example, 10k pictures of 1k persons.
        
        Arguments:
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor images, of shape (None, 128)
                positive -- the encodings for the positive images, of shape (None, 128)
                negative -- the encodings for the negative images, of shape (None, 128)
        
        Returns:
        loss -- real number, value of the loss
        """
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        
        #(≈ 4 lines)
        # Step 1: Compute the (encoding) distance between the anchor and the positive
        pos = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
        # Step 2: Compute the (encoding) distance between the anchor and the negative
        neg = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos, neg), alpha)
        print(f"anchor: {anchor}, pos: {pos.shape}, neg: {neg.shape}, basic_loss: {basic_loss.shape}, alpha: {alpha}")
        print(f"pos: {pos}, neg: {neg}, basic_loss: {basic_loss}")
        print(f"max: {tf.maximum(basic_loss, 0.0)}")
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
        print(f"loss: {loss}")
        return loss
    #tf.keras.backend.set_image_data_format('channels_last')
    def _img_to_encoding(self, image_path:str):
        """
        Generate one encoding vector for each person by running the forward propagation of the model on the specified image.
        """
        img = load_img(image_path, target_size=(299, 299))
        data = preprocess_input(numpy.expand_dims(img, axis=0))
        embedding = self._model.predict_on_batch(data)
        return embedding / numpy.linalg.norm(embedding, ord=2)
    
    def triplet_loss_test(self):
        print(f"\n=== {self.triplet_loss_test.__name__} ===")
        y_pred_perfect = ([[1., 1.]], [[1., 1.]], [[1., 1.,]])
        loss = self._triplet_loss(y_pred_perfect, 5)
        assert loss == 5, "Wrong value. Did you add the alpha to basic_loss?"
        y_pred_perfect = ([[1., 1.]],[[1., 1.]], [[0., 0.,]])
        loss = self._triplet_loss(y_pred_perfect, 3)
        assert loss == 1., "Wrong value. Check that pos_dist = 0 and neg_dist = 2 in this example"
        y_pred_perfect = ([[1., 1.]],[[0., 0.]], [[1., 1.,]])
        loss = self._triplet_loss(y_pred_perfect, 0)
        assert loss == 2., "Wrong value. Check that pos_dist = 2 and neg_dist = 0 in this example"
        y_pred_perfect = ([[0., 0.]],[[0., 0.]], [[0., 0.,]])
        loss = self._triplet_loss(y_pred_perfect, -2)
        assert loss == 0, "Wrong value. Are you taking the maximum between basic_loss and 0?"
        y_pred_perfect = ([[1., 0.], [1., 0.]],[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]])
        loss = self._triplet_loss(y_pred_perfect, 3)
        assert loss == 2., "Wrong value. Are you applying tf.reduce_sum to get the loss?"
        y_pred_perfect = ([[1., 1.], [2., 0.]], [[0., 3.], [1., 1.]], [[1., 0.], [0., 1.,]])
        loss = self._triplet_loss(y_pred_perfect, 1)
        if (loss == 4.):
            raise Exception('Perhaps you are not using axis=-1 in reduce_sum?')
        assert loss == 5, "Wrong value. Check your implementation"    
        print(f"{bcolors.OKGREEN}All tests passed!{bcolors.DEFAULT}")

    def SimilarityTests(self):
        print(f"\n=== {self.SimilarityTests.__name__} ===")
        for n in self._names:
            path = f"images/{n}.png" if n == "danielle" else f"images/{n}.jpg"
            distance, door_open_flag = faceRecognition.Verify(path, n)
            assert numpy.isclose(distance, 0.0)
            assert door_open_flag, f"{bcolors.FAIL}Door should be opened for {n} - ({distance} {door_open_flag}){bcolors.DEFAULT}"
            dist, identity = faceRecognition.Who_Is_It(path)
            assert numpy.isclose(dist, 0.0)
            assert identity == n, f"identity: {identity}, distance: {dist}"
        print(f"{bcolors.OKGREEN}All tests passed!{bcolors.DEFAULT}")
    
def CameraPicturesTests():
    print(f"\n=== {CameraPicturesTests.__name__} ===")

    distance, door_open_flag = faceRecognition.Verify("images/camera_0.jpg", "younes")
    assert door_open_flag, f"{bcolors.FAIL}Door should be opened for younes - ({distance} {door_open_flag}){bcolors.DEFAULT}"

    distance, door_open_flag = faceRecognition.Verify("images/camera_1.jpg", "bertrand")
    assert door_open_flag, f"{bcolors.FAIL}Door should be opened for bertrand - ({distance} {door_open_flag}){bcolors.DEFAULT}"

    distance, door_open_flag = faceRecognition.Verify("images/camera_2.jpg", "benoit")
    assert door_open_flag, f"{bcolors.FAIL}Door should be opened for benoit - ({distance} {door_open_flag}){bcolors.DEFAULT}"

    distance, door_open_flag = faceRecognition.Verify("images/camera_3.jpg", "bertrand")
    assert door_open_flag, f"{bcolors.FAIL}Door should be opened for bertrand - ({distance} {door_open_flag}){bcolors.DEFAULT}"

    distance, door_open_flag = faceRecognition.Verify("images/camera_4.jpg", "dan")
    assert door_open_flag, f"{bcolors.FAIL}Door should be opened for dan - ({distance} {door_open_flag}){bcolors.DEFAULT}"

    distance, door_open_flag = faceRecognition.Verify("images/camera_5.jpg", "arnaud")
    assert door_open_flag, f"{bcolors.FAIL}Door should be opened for arnaud - ({distance} {door_open_flag}){bcolors.DEFAULT}"

    dist, identity = faceRecognition.Who_Is_It("images/camera_0.jpg")
    assert identity == 'younes', f"identity: {identity}, distance: {dist}"

    dist, identity = faceRecognition.Who_Is_It("images/camera_1.jpg")
    assert identity == 'bertrand', f"identity: {identity}, distance: {dist}"

    dist, identity = faceRecognition.Who_Is_It("images/camera_2.jpg")
    assert identity == 'benoit', f"identity: {identity}, distance: {dist}"

    dist, identity = faceRecognition.Who_Is_It("images/camera_3.jpg")
    assert identity == 'bertrand', f"identity: {identity}, distance: {dist}"

    dist, identity = faceRecognition.Who_Is_It("images/camera_4.jpg")
    assert identity == 'dan', f"identity: {identity}, distance: {dist}"

    dist, identity = faceRecognition.Who_Is_It("images/camera_5.jpg")
    assert identity == 'arnaud', f"identity: {identity}, distance: {dist}"

    distance, door_open_flag = faceRecognition.Verify("images/camera_1.jpg", "kian")
    assert not door_open_flag, f"{bcolors.FAIL}Door should NOT be opened for kian - ({distance} {door_open_flag}){bcolors.DEFAULT}"

    # Test 3 with Younes pictures 
    dist, identity = faceRecognition.Who_Is_It("images/younes.jpg")
    assert numpy.isclose(dist, 0.0)
    assert identity == 'younes', f"identity: {identity}, distance: {dist}"
    print(f"{bcolors.OKGREEN}All tests passed!{bcolors.DEFAULT}")

if __name__ == "__main__":
    faceRecognition = FaceRecognition(0.1)
    faceRecognition.triplet_loss_test()
    faceRecognition.SimilarityTests()
    CameraPicturesTests()
