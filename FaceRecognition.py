import os, numpy, pandas as pd, tensorflow as tf, PIL
from numpy import genfromtxt
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, BatchNormalization, MaxPooling2D, AveragePooling2D, Concatenate, Lambda, Flatten, Dense, Layer
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

class FaceRecognition():
    _model: Model = None
    _threshold: float = None
    _database = {}
    def __init__(self, threshold:bool):
        self._threshold = threshold
        self.BuildModel()
        self.PrepareData()

    def BuildModel(self):
        """
        This network uses 160x160 dimensional RGB images as its input. Specifically, a face image (or batch of  𝑚 face images) as a tensor of shape  (𝑚,𝑛𝐻,𝑛𝑊,𝑛𝐶)=(𝑚,160,160,3)
        The input images are originally of shape 96x96, thus, you need to scale them to 160x160. This is done in the self._img_to_encoding() function.
        The output is a matrix of shape  (𝑚,128) that encodes each input face image into a 128-dimensional vector.
        By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. You then use the encodings to compare two face images as follows:
        """
        with open('./models/keras-facenet-h5/model.json', 'r', newline='') as f:
            loaded_model_json = f.read()
            self._model = model_from_json(loaded_model_json)
        self._model.load_weights('./models/keras-facenet-h5/model.h5')
        print(f"Model input: {self._model.inputs}")
        print(f"Model output: {self._model.outputs}")

    def PrepareData(self):
        self._database["danielle"] = self._img_to_encoding("images/danielle.png")
        self._database["younes"] = self._img_to_encoding("images/younes.jpg")
        self._database["tian"] = self._img_to_encoding("images/tian.jpg")
        self._database["andrew"] = self._img_to_encoding("images/andrew.jpg")
        self._database["kian"] = self._img_to_encoding("images/kian.jpg")
        self._database["dan"] = self._img_to_encoding("images/dan.jpg")
        self._database["sebastiano"] = self._img_to_encoding("images/sebastiano.jpg")
        self._database["bertrand"] = self._img_to_encoding("images/bertrand.jpg")
        self._database["kevin"] = self._img_to_encoding("images/kevin.jpg")
        self._database["felix"] = self._img_to_encoding("images/felix.jpg")
        self._database["benoit"] = self._img_to_encoding("images/benoit.jpg")
        self._database["arnaud"] = self._img_to_encoding("images/arnaud.jpg")

    def Verify(self, image_path, identity):
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
        #print(f"encoding: {encoding}, identity: {database[identity]}, diff: {encoding - database[identity]}")
        dist1 = encoding -  self._database[identity]
        dist = numpy.linalg.norm(encoding - self._database[identity], ord=2)
        print(f"distance: {dist1.shape}, {encoding - self._database[identity]}, ord=2: {dist}")
        # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
        if dist < self._threshold:
            print("It's " + str(identity) + ", welcome in!")
            door_open = True
        else:
            print("It's not " + str(identity) + ", please go away")
            door_open = False
        return dist, door_open

    def Who_Is_It(self, image_path):
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
            print("Not in the database.")
        else:
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))
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
    def _img_to_encoding(self, image_path):
        """
        Generate one encoding vector for each person by running the forward propagation of the model on the specified image.
        """
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img = numpy.around(numpy.array(img) / 255.0, decimals=12)
        x_train = numpy.expand_dims(img, axis=0)
        embedding = self._model.predict_on_batch(x_train)
        return embedding / numpy.linalg.norm(embedding, ord=2)
    
    def triplet_loss_test(self):
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

if __name__ == "__main__":
    faceRecognition = FaceRecognition(0.7)
    faceRecognition.triplet_loss_test()
    distance, door_open_flag = faceRecognition.Verify("images/camera_0.jpg", "younes")
    assert numpy.isclose(distance, 0.5992949), "Distance not as expected"
    assert door_open_flag, "Door should be opened for younes"
    print("(", distance, ",", door_open_flag, ")")
    distance, door_open_flag = faceRecognition.Verify("images/camera_1.jpg", "kian")
    #assert numpy.isclose(distance, 0.5992949), "Distance not as expected"
    assert not door_open_flag, "Door should NOT be opened for kian"
    print("(", distance, ",", door_open_flag, ")")

    # Test 2 with Younes pictures 
    dist, identity = faceRecognition.Who_Is_It("images/camera_0.jpg")
    assert numpy.isclose(dist, 0.5992946)
    assert identity == 'younes'

    # Test 3 with Younes pictures 
    dist, identity = faceRecognition.Who_Is_It("images/younes.jpg")
    assert numpy.isclose(dist, 0.0)
    assert identity == 'younes'