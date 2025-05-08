from pathlib import Path
import numpy, math, tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import numpy.lib.recfunctions as reconcile
import matplotlib.pyplot as plt
from utils.TensorModelPlot import PlotModelHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())
# https://realpython.com/generative-adversarial-networks/
class Discriminator():
    _model = None
    def __init__(self):
        """
        Multilayer Perceptron NN defined in a sequential way using models.Sequential()
        The input is two-dimensional, and the first hidden layer is composed of 256 neurons with ReLU activation.
        The second and third hidden layers are composed of 128 and 64 neurons, respectively, with ReLU activation.
        The output is composed of a single neuron with sigmoidal activation to represent a probability.
        Pytorch:
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        https://stackoverflow.com/questions/66626700/difference-between-tensorflows-tf-keras-layers-dense-and-pytorchs-torch-nn-lin
        """
        self._model = models.Sequential()
        self._model.add(layers.Dense(256, input_shape=(2,), activation='relu'))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(128, input_shape=(256,), activation='relu'))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(64, input_shape=(128,), activation='relu'))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(1, input_shape=(64,), activation='sigmoid'))
        self._model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
                metrics=['accuracy'])
        
    def Train(self, data, batch_size):
        """
        WIP. Need to adapt to tensorflow
        """
        # Data for training the discriminator
        real_samples_labels = numpy.ones((batch_size, 1))
        latent_space_samples = rng.random((batch_size, 2))
        generated_samples = generator(latent_space_samples) # This is crap. Generator constructor does not take any input!
        generated_samples_labels = numpy.zeros((batch_size, 1))
        all_samples = numpy.concatenate((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )
        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 2))        
        history = self._model.fit(
            x_train, 
            y_train, 
            epochs=25,
            validation_data=(x_test, y_test)
        )
        print("Model Summary:") # This has to be done AFTER fit as there is no explicit Input layer added
        self._model.summary()
        train_loss, train_accuracy = self._model.evaluate(x_train, y_train, verbose=2)
        test_loss, test_accuracy = self._model.evaluate(x_test, y_test, verbose=2)
        print(f'Training accuracy: {train_accuracy:.4f}, loss: {train_loss:.4f}')
        print(f'Testing accuracy: {test_accuracy:.4f}, loss: {test_loss:.4f}')
        PlotModelHistory("Discriminator", history)

    def forward(self, input):
        return self._model(input)
    
class Generator():
    _model = None
    def __init__(self):
        """
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )        
        """
        self._model = models.Sequential()
        self._model.add(layers.Dense(16, input_shape=(2,), activation='relu'))
        self._model.add(layers.Dense(32, input_shape=(16,), activation='relu'))
        self._model.add(layers.Dense(2, input_shape=(32,)))
        self._model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
                metrics=['accuracy'])
        history = self._model.fit(
            x_train, 
            y_train, 
            epochs=25,
            validation_data=(x_test, y_test)
        )
        print("Model Summary:") # This has to be done AFTER fit as there is no explicit Input layer added
        self._model.summary()
        train_loss, train_accuracy = self._model.evaluate(x_train, y_train, verbose=2)
        test_loss, test_accuracy = self._model.evaluate(x_test, y_test, verbose=2)
        print(f'Training accuracy: {train_accuracy:.4f}, loss: {train_loss:.4f}')
        print(f'Testing accuracy: {test_accuracy:.4f}, loss: {test_loss:.4f}')
        PlotModelHistory("Generator", history)

    def Train(self, data, batch_size):
        """
        WIP. Need to adapt to tensorflow
        """
        # Data for training the generator
        latent_space_samples = rng.random((batch_size, 2))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples) # This is crap. Generator constructor does not take any input!
        output_discriminator_generated = discriminator(generated_samples)  # This is crap. Generator constructor does not take any input!
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

    def forward(self, input):
        return self._model(input)
    
def PrepareTrainingData(size: int):
    data = numpy.zeroes((size, 2))
    data[:,0] = 2 * math.pi * rng.random(size)
    data[:,1] = math.sin(data[:,0])
    labels = numpy.zeroes(size)
    train = [(data[i], labels[i]) for i in range(size)]
    # result = tf.multiply(tf.convert_to_tensor(array1), tf.convert_to_tensor(array2))
    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    tf.data.Dataset.from_tensor_slices(train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

if __name__ == "__main__":