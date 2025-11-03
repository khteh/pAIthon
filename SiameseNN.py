import argparse, numpy, pandas as pd, tensorflow as tf
from pathlib import Path
from tensorflow.keras.utils import plot_model
from tensorflow.math import l2_normalize
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Lambda, Concatenate, Dense, Input, LSTM, Embedding, TextVectorization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from utils.TrainingMetricsPlot import PlotModelHistory
from utils.ConfusionMatrix import ConfusionMatrix
from utils.TrainingUtils import CreateTensorBoardCallback, CreateCircuitBreakerCallback
class SiameseNN():
    """
    Siamese NN text similarity classifier.
    """
    _path:str = None
    _model_path:str = None
    _embedding_dimension: int = None
    _data_train = None
    _data_test = None
    _Q1_train = None
    _Q2_train = None
    _Q1_test = None
    _Q2_test = None
    _y_test = None
    _train_Q1 = None
    _train_Q2 = None
    _val_Q1 = None
    _val_Q2 = None
    _train_dataset = None
    _val_dataset = None
    _test_dataset = None
    _text_vectorizer = None
    _vocab = None
    _model = None
    _circuit_breaker = None
    _batch_size:int = None
    _learning_rate: float = None
    def __init__(self, path: str, model_path:str, embedding_dim:int, batch_size:int, learning_rate:float):
        self._path = path
        self._model_path = model_path
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._embedding_dimension = embedding_dim
        self._PrepareData()
        self._circuit_breaker = CreateCircuitBreakerCallback("val_loss", "min", 5)
        #if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
        #    print(f"Using saved model {self._model_path}...")
        #    self._model = load_model(self._model_path) https://github.com/tensorflow/tensorflow/issues/102475

    def BuildTrainModel(self, epochs:int, retrain: bool = False):
        new_model = not self._model
        if not self._model:
            branch = Sequential([
                self._text_vectorizer,
                Embedding(len(self._vocab), self._embedding_dimension, name="Embedding"),
                LSTM(self._embedding_dimension, return_sequences=True, name="LSTM"),
                GlobalAveragePooling1D(name="mean"),
                # This applies L2 normalization directly to the output of the preceding layer. It ensures that the L2 norm of the output vector (or along a specified axis) becomes 1. This is a transformation of the activations.
                Lambda(l2_normalize, name="out")
            ], name="SiameseSequential")
            # Define both inputs. Remember to call then 'input_1' and 'input_2' using the `name` parameter. 
            # Be mindful of the data type and size
            input1 = Input(shape=(1,), dtype=tf.string, name='input_1')
            input2 = Input(shape=(1,), dtype=tf.string, name='input_2')
            # Define the output of each branch of your Siamese network. Remember that both branches have the same coefficients, 
            # but they each receive different inputs.
            branch1 = branch(input1)
            branch2 = branch(input2)
            # Define the Concatenate layer. You should concatenate columns, you can fix this using the `axis`parameter. 
            # This layer is applied over the outputs of each branch of the Siamese network
            conc = Concatenate(axis=-1, name='CosineSimilarity')([branch1, branch2]) 
            self._model = Model(inputs=[input1, input2], outputs=conc, name="SiameseModel")
            self._model.build(input_shape=None)
            self._model.summary()
            self._model.get_layer(name='SiameseSequential').summary()        
            plot_model(
                self._model,
                to_file="output/SiameseNN.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
            # Compile the model
            self._model.compile(loss=self._TripletLoss, optimizer = Adam(learning_rate=self._learning_rate))
        if new_model or retrain:
            # Train the model 
            tensorboard = CreateTensorBoardCallback("SiameseNN") # Create a new folder with current timestamp
            history = self._model.fit(self._train_dataset, epochs = epochs, shuffle=True, validation_data = self._val_dataset, validation_freq=1, callbacks=[tensorboard, self._circuit_breaker]) # shuffle: Boolean, whether to shuffle the training data before each epoch. This argument is ignored when x is a generator or a tf.data.Dataset.
            PlotModelHistory("Siamese NN", history)
            #if self._model_path:
            #    self._model.save(self._model_path) https://github.com/tensorflow/tensorflow/issues/102475
            #    print(f"Model saved to {self._model_path}.")
        
    def Evaluate(self, threshold, verbose=True):
        """Function to test the accuracy of the model.

        Args:
            test_Q1 (numpy.ndarray): Array of Q1 questions. Each element of the array would be a string.
            test_Q2 (numpy.ndarray): Array of Q2 questions. Each element of the array would be a string.
            y_test (numpy.ndarray): Array of actual target.
            threshold (float): Desired threshold
            model (tensorflow.Keras.Model): The Siamese model.
            batch_size (int, optional): Size of the batches. Defaults to 64.

        Returns:
            float: Accuracy of the model
            numpy.array: confusion matrix
        """
        y_pred = []
        
        ### START CODE HERE ###
        #print(f"test_gen: {test_gen}")
        pred = self._model.predict(self._test_dataset)
        #print(f"pred: {pred.shape}")
        _, n_feat = pred.shape
        #print(f"n_feat: {n_feat}")
        v1 = pred[:,:int(n_feat/2)] # Extract v1 from out
        v2 = pred[:,int(n_feat/2):] # Extract v2 from out    
        print(f"v1: {v1.shape}, v2: {v2.shape}")
        # Compute the cosine similarity. Using `tf.math.reduce_sum`. 
        # Don't forget to use the appropriate axis argument.
        d  = tf.math.reduce_sum(v1 * v2, axis=1)
        print(f"d: {d.shape}")
        # Check if d>threshold to make predictions
        y_pred = tf.cast(d > threshold, tf.float64)
        
        # take the average of correct predictions to get the accuracy
        correct_pred = y_pred == self._y_test
        #print(f"y_pred: {y_pred}, correct_pred: {correct_pred} type: {type(correct_pred)}")
        accuracy = tf.reduce_sum(tf.cast(y_pred == self._y_test, tf.int64)) / self._y_test.shape[0]
        # compute the confusion matrix using `tf.math.confusion_matrix`
        print(f"y_pred: {y_pred.shape}, y_test: {self._y_test.shape}")
        cm = tf.math.confusion_matrix(self._y_test, y_pred)
        print("Accuracy", accuracy.numpy())
        print(f"Confusion matrix:\n{cm.numpy()}")
        #ConfusionMatrix(self._y_test, y_pred, "Siamese NN")

    def Predict(self, question1, question2, threshold, verbose=False):
        """Function for predicting if two questions are duplicates.

        Args:
            question1 (str): First question.
            question2 (str): Second question.
            threshold (float): Desired threshold.
            model (tensorflow.keras.Model): The Siamese model.
            data_generator (function): Data generator function. Defaults to data_generator.
            verbose (bool, optional): If the results should be printed out. Defaults to False.

        Returns:
            bool: True if the questions are duplicates, False otherwise.
        """
        generator = tf.data.Dataset.from_tensor_slices((([question1], [question2]),None)).batch(batch_size=1).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Call the predict method of your model and save the output into v1v2
        pred = self._model.predict(generator)
        _, n_feat = pred.shape
        #print(f"n_feat: {n_feat}")
        v1 = pred[:,:int(n_feat/2)] # Extract v1 from out
        v2 = pred[:,int(n_feat/2):] # Extract v2 from out    
        # Take the dot product to compute cos similarity of each pair of entries, v1, v2
        # Since v1 and v2 are both vectors, use the function tf.math.reduce_sum instead of tf.linalg.matmul
        d  = tf.math.reduce_sum(v1 * v2, axis=1)
        # Is d greater than the threshold?
        y_pred = (d > threshold)[0]
        if(verbose):
            print("Q1  = ", question1, "\nQ2  = ", question2)
            print("d   = ", d.numpy())
            print("res = ", y_pred.numpy())
        return y_pred.numpy()
    
    def _PrepareData(self):
        """
        Build question pairs which are duplicates to train the model.
        2 sets of questions as input to the Siamese NN, assuming that q1 in the first set is a duplicate of q2 in the second set. All other questions in the second set are not duplicates of q1.
        The test set uses the original pairs of questions and the label indicating if they are duplicates.
        Select only duplicate questions from the training set. Firstly, find the indices with duplicate questions.
        Start by identifying the indices in the training dataset which correspond to duplicate questions.
        """
        data = pd.read_csv(self._path)
        N = len(data)
        print('Number of question pairs: ', N)
        data.head()       
        N_train = 300000
        N_test = 10240
        self._data_train = data[:N_train]
        self._data_test = data[N_train:N_train + N_test]
        print("Train set:", len(self._data_train), "Test set:", len(self._data_test))
        del (data)  # remove to free memory
        td_index = self._data_train['is_duplicate'] == 1
        td_index = [i for i, x in enumerate(td_index) if x]
        print('Number of duplicate questions: ', len(td_index))
        print('Indexes of first ten duplicate questions:', td_index[:10])
        print(self._data_train['question1'][5])
        print(self._data_train['question2'][5])
        print('is_duplicate: ', self._data_train['is_duplicate'][5])
        # Keep only the rows in the original training set which correspond to the rows where td_index = True
        self._Q1_train = numpy.array(self._data_train['question1'][td_index])
        self._Q2_train = numpy.array(self._data_train['question2'][td_index])

        self._Q1_test = numpy.array(self._data_test['question1'])
        self._Q2_test = numpy.array(self._data_test['question2'])
        self._y_test  = numpy.array(self._data_test['is_duplicate'])        

        print('TRAINING QUESTIONS:\n')
        print('Question 1: ', self._Q1_train[0])
        print('Question 2: ', self._Q2_train[0], '\n')
        print('Question 1: ', self._Q1_train[5])
        print('Question 2: ', self._Q2_train[5], '\n')

        print('TESTING QUESTIONS:\n')
        print('Question 1: ', self._Q1_test[0])
        print('Question 2: ', self._Q2_test[0], '\n')
        print('is_duplicate =', self._y_test[0], '\n')

        # Splitting the data
        cut_off = int(len(self._Q1_train) * 0.8)
        self._train_Q1, self._train_Q2 = self._Q1_train[:cut_off], self._Q2_train[:cut_off]
        self._val_Q1, self._val_Q2 = self._Q1_train[cut_off:], self._Q2_train[cut_off:]
        print('Number of duplicate questions: ', len(self._Q1_train))
        print("The length of the training set is:  ", len(self._train_Q1))
        print("The length of the validation set is: ", len(self._val_Q1))        

        self._text_vectorizer = TextVectorization(output_mode='int',split='whitespace', standardize='strip_punctuation')
        self._text_vectorizer.adapt(numpy.concatenate((self._Q1_train, self._Q2_train)))
        self._vocab = self._text_vectorizer.get_vocabulary()
        self._train_dataset = tf.data.Dataset.from_tensor_slices(((self._train_Q1, self._train_Q2),tf.constant([1]*len(self._train_Q1))))
        self._val_dataset = tf.data.Dataset.from_tensor_slices(((self._val_Q1, self._val_Q2),tf.constant([1]*len(self._val_Q1))))
        self._train_dataset = self._train_dataset.shuffle(len(self._train_Q1),
                                                seed=7, 
                                                reshuffle_each_iteration=True).batch(batch_size=self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self._val_dataset = self._val_dataset.shuffle(len(self._val_Q1), 
                                        seed=7,
                                        reshuffle_each_iteration=True).batch(batch_size=self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self._test_dataset = tf.data.Dataset.from_tensor_slices(((self._Q1_test, self._Q2_test),None)).batch(batch_size=self._batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def _TripletLoss(self, labels, out, margin=0.25):
        _, embedding_size = out.shape # get embedding size
        v1 = out[:,:int(embedding_size/2)] # Extract v1 from out
        v2 = out[:,int(embedding_size/2):] # Extract v2 from out
        return self._TripletLossFn(v1, v2, margin=margin)
    
    def _TripletLossFn(self, v1, v2,  margin=0.25):
        """Custom Loss function.

        Args:
            v1 (numpy.ndarray or Tensor): Array with dimension (batch_size, model_dimension) associated to Q1.
            v2 (numpy.ndarray or Tensor): Array with dimension (batch_size, model_dimension) associated to Q2.
            margin (float, optional): Desired margin. Defaults to 0.25.

        Returns:
            triplet_loss (numpy.ndarray or Tensor)
        """
        #print(f"\nv1: {v1.shape}")
        #print(f"\nv2: {v2.shape}")
        # use `tf.linalg.matmul` to take the dot product of the two batches. 
        # Don't forget to transpose the second argument using `transpose_b=True`
        scores = tf.linalg.matmul(v2, v1, transpose_b=True)
        # calculate new batch size and cast it as the same datatype as scores. 

        batch_size = tf.cast(tf.shape(v1)[0], scores.dtype) 
        #print(f"batch_size: {batch_size}")
        # use `tf.linalg.diag_part` to grab the cosine similarity of all positive examples
        positive = tf.linalg.diag_part(scores)
        #print(f"\nscores: {scores}")
        #print(f"\npositive: {positive}")
        # subtract the diagonal from scores. You can do this by creating a diagonal matrix with the values 
        # of all positive examples using `tf.linalg.diag`
        #print(f"\ntf.linalg.diag(scores): {tf.linalg.diag(scores)}")
        #print(f"\ntf.linalg.diag(positive): {tf.linalg.diag(positive)}")
        diagonal = tf.linalg.diag(positive)
        #print(f"\ndiagonal: {diagonal}")
        negative_zero_on_duplicate = tf.math.subtract(scores, tf.linalg.diag(positive))
        #print(f"\nnegative_zero_on_duplicate: {negative_zero_on_duplicate}")
        # use `tf.math.reduce_sum` on `negative_zero_on_duplicate` for `axis=1` and divide it by `(batch_size - 1)`
        mean_negative = tf.math.divide(tf.math.reduce_sum(negative_zero_on_duplicate, axis=1), (batch_size - 1))
        # create a composition of two masks: 
        # the first mask to extract the diagonal elements, 
        # the second mask to extract elements in the negative_zero_on_duplicate matrix that are larger than the elements in the diagonal 
        diagonal_mask = tf.eye(batch_size) == 1
        #print(f"\ndiagonal_mask: {diagonal_mask}")
        #print(f"\ntf.expand_dims(positive, 1): {tf.expand_dims(positive, 1)}")
        larger_than_diagonal = negative_zero_on_duplicate > tf.expand_dims(positive, 1)
        #print(f"\nlarger_than_diagonal: {larger_than_diagonal}")
        mask_exclude_positives = tf.cast((tf.eye(batch_size) == 1)|(negative_zero_on_duplicate > tf.expand_dims(positive, 1)), scores.dtype)
        #print(f"\nmask_exclude_positives: {mask_exclude_positives}")
        # multiply `mask_exclude_positives` with 2.0 and subtract it out of `negative_zero_on_duplicate`
        negative_without_positive = tf.math.subtract(negative_zero_on_duplicate,  mask_exclude_positives * 2.0)
        #print(f"\nnegative_without_positive: {negative_without_positive}")
        # take the row by row `max` of `negative_without_positive`. 
        # Hint: `tf.math.reduce_max(negative_without_positive, axis = None
        closest_negative = tf.math.reduce_max(negative_without_positive, axis=1)
        # compute `tf.maximum` among 0.0 and `A`
        # A = subtract `positive` from `margin` and add `closest_negative`
        #print(f"\nmean_negative: {mean_negative}")
        #print(f"\nclosest_negative: {closest_negative}")
        A = margin - positive + closest_negative
        L1 = tf.maximum(0.0, A)
        # compute `tf.maximum` among 0.0 and `B`
        # B = subtract `positive` from `margin` and add `mean_negative` 
        B = margin - positive + mean_negative
        L2 = tf.maximum(0.0, B)
        # add the two losses together and take the `tf.math.reduce_sum` of it
        L = tf.math.reduce_sum(L1 + L2)
        #print(f"\n{L1} + {L2} = {L}")
        return L
    
if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Siamese NN text similarity classifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    siamese = SiameseNN("data/questions.csv", "models/SiameseNN.keras", 128, 512, 0.001)
    siamese.BuildTrainModel(30, args.retrain)
    siamese.Evaluate(0.7)
    question1 = "When will I see you?"
    question2 = "When can I see you again?"
    # 1 means it is duplicated, 0 otherwise
    prediction = siamese.Predict(question1 , question2, 0.7, verbose = True)
    assert prediction == True
    print(f"prediction: {prediction}")

    question1 = "Do they enjoy eating the dessert?"
    question2 = "Do they like hiking in the desert?"
    # 1 means it is duplicated, 0 otherwise
    prediction = siamese.Predict(question1 , question2, 0.7, verbose = True)
    assert prediction == False
    print(f"prediction: {prediction}")