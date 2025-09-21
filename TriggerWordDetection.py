import numpy, random, sys, io, os, glob
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from utils.TensorModelPlot import PlotModelHistory

class TrigerWordDetection():
    _path:str = None
    _dev_path:str = None
    _activates = None
    _negatives = None
    _backgrounds = None
    _Tx: int = None
    _Ty: int = None
    _n_freq: int = None
    _X = None
    _Y = None
    _X_dev = None
    _Y_dev = None
    _samples:int = None
    _model: Sequential = None
    _learning_rate: float = None
    _beta_1: float = None
    _beta_2: float = None
    _batch_size: int = None
    _epochs:int = None
    _chime_file = "data/chime.wav"

    def __init__(self, path:str, devpath:str, tx:int, ty:int, n_freq:int, samples:int, learning_rate:float, beta1:float, beta2:float, batchsize: int, epochs:int):
        self._Tx = tx
        self._Ty = ty
        self._n_freq = n_freq
        self._samples = samples
        self._path = path
        self._dev_path = devpath
        self._learning_rate = learning_rate
        self._beta_1 = beta1
        self._beta_2 = beta2
        self._batch_size = batchsize
        self._epochs = epochs
        self._PrepareData()

    def _PrepareData(self):
        # Load audio segments using pydub 
        self._load_raw_audio()
        self._X = []
        self._Y = []
        for i in range(0, self._samples):
            if i%10 == 0:
                print(i)
            x, y = self.create_training_example(i % 2)
            self._X.append(x.swapaxes(0,1))
            self._Y.append(y.swapaxes(0,1))
        self._X = numpy.array(self._X)
        self._Y = numpy.array(self._Y)
        if self._dev_path:
            # Load preprocessed dev set examples
            self._X_dev = numpy.load(f"{self._dev_path}/X_dev.npy")
            self._Y_dev = numpy.load(f"{self._dev_path}/Y_dev.npy")            
        print("background len should be 10,000, since it is a 10 sec clip\n" + str(len(self._backgrounds[0])),"\n")
        print("activate[0] len may be around 1000, since an `activate` audio clip is usually around 1 second (but varies a lot) \n" + str(len(self._activates[0])),"\n")
        print("activate[1] len: different `activate` clips can have different lengths\n" + str(len(self._activates[1])),"\n")

    def get_random_time_segment(self, segment_ms):
        """
        Gets a random time segment of duration segment_ms in a 10,000 ms audio clip onto which we can insert an audio clip of duration segment_ms.
        
        Arguments:
        segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
        
        Returns:
        segment_time -- a tuple of (segment_start, segment_end) in ms
        """
        segment_start = numpy.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
        segment_end = segment_start + segment_ms - 1
        return (segment_start, segment_end)
    
    def IsOverlapping(self, segment_time, previous_segments):
        """
        Checks if the time of a segment overlaps with the times of existing segments.
        
        Arguments:
        segment_time -- a tuple of (segment_start, segment_end) for the new segment
        previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
        
        Returns:
        True if the time segment overlaps with any of the existing segments, False otherwise
        """
        segment_start, segment_end = segment_time
        # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
        for previous_start, previous_end in previous_segments:
            if segment_start <= previous_end and segment_end >= previous_start:
                return True
        return False
    
    def InsertAudioClip(self, background_index:int, audio_index:int, previous_segments):
        audio_clip, segment_time = self._insert_audio_clip(self._backgrounds[background_index], self._activates[audio_index], previous_segments)
        duration = segment_time[1] - segment_time[0]
        print(f"duration: {duration}")
        assert audio_clip
        if duration:
            assert duration + 1 == len(self._activates[audio_index]) , "The segment length must match the audio clip length"
            assert audio_clip != self._backgrounds[background_index] , "The audio clip must be different than the pure background"
        else:
            assert audio_clip == self._backgrounds[background_index], "output audio clip must be exactly the same input background"
        return audio_clip, segment_time

    def insert_ones(self, y, segment_end_ms):
        """
        Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
        should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
        50 following labels should be ones.
        
        Arguments:
        y -- numpy array of shape (1, Ty), the labels of the training example
        segment_end_ms -- the end time of the segment in ms
        
        Returns:
        y -- updated labels
        """
        _, Ty = y.shape
        
        # duration of the background (in terms of spectrogram time-steps)
        segment_end_y = int(segment_end_ms * Ty / 10000.0)
        
        if segment_end_y < Ty:
            # Add 1 to the correct index in the background label (y)
            for i in range(segment_end_y + 1, min(segment_end_y + 1 + 50, Ty)):
                if i < Ty:
                    y[0, i] = 1
        return y
        
    def create_training_example(self, background_index:int):
        """
        Creates a training example with a given background, activates, and negatives.
        
        Arguments:
        background -- a 10 second background audio recording
        activates -- a list of audio segments of the word "activate"
        negatives -- a list of audio segments of random words that are not "activate"
        Ty -- The number of time steps in the output

        Returns:
        x -- the spectrogram of the training example
        y -- the label at each time step of the spectrogram
        """
        background = self._backgrounds[background_index]
        # Make background quieter
        background = background - 20

        ### START CODE HERE ###
        # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
        y = numpy.zeros((1, self._Ty))

        # Step 2: Initialize segment times as empty list (≈ 1 line)
        previous_segments = []
        ### END CODE HERE ###
        
        # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
        number_of_activates = numpy.random.randint(0, 5)
        random_indices = numpy.random.randint(len(self._activates), size=number_of_activates)
        random_activates = [self._activates[i] for i in random_indices]
        
        ### START CODE HERE ### (≈ 3 lines)
        # Step 3: Loop over randomly selected "activate" clips and insert in background
        for one_random_activate in random_activates:
            # Insert the audio clip on the background
            # def insert_audio_clip(background, audio_clip, previous_segments):
            background, segment_time = self._insert_audio_clip(background, one_random_activate, previous_segments)
            # Retrieve segment_start and segment_end from segment_time
            segment_start, segment_end = segment_time[0], segment_time[1]
            # Insert labels in "y" at segment_end
            y = self.insert_ones(y, segment_end)
        ### END CODE HERE ###

        # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
        number_of_negatives = numpy.random.randint(0, 3)
        random_indices = numpy.random.randint(len(self._negatives), size=number_of_negatives)
        random_negatives = [self._negatives[i] for i in random_indices]

        ### START CODE HERE ### (≈ 2 lines)
        # Step 4: Loop over randomly selected negative clips and insert in background
        for random_negative in random_negatives:
            # Insert the audio clip on the background 
            background, _ = self._insert_audio_clip(background, random_negative, previous_segments)
        ### END CODE HERE ###
        
        # Standardize the volume of the audio clip 
        background = self._match_target_amplitude(background, -20.0)

        # Export new training example 
        file_handle = background.export("train" + ".wav", format="wav")
        
        # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
        x = self._graph_spectrogram("train.wav")
        return x, y
   
    def BuildModel(self):
        """
        build a network that will ingest a spectrogram and output a signal when it detects the trigger word. This network will use 4 layers:

        * A convolutional layer
        * Two GRU layers
        * A dense layer. 

        1D convolutional layer
        One key layer of this model is the 1D convolutional step (near the bottom of Figure 3).

        - It inputs the 5511 step spectrogram. Each step is a vector of 101 units.
        - It outputs a 1375 step output
        - This output is further processed by multiple layers to get the final  𝑇𝑦=1375 step output.
        - This 1D convolutional layer plays a role similar to the 2D convolutions you saw in Course 4, of extracting low-level features and then possibly generating an output of a smaller dimension.
        - Computationally, the 1-D conv layer also helps speed up the model because now the GRU can process only 1375 timesteps rather than 5511 timesteps.

        GRU, dense and sigmoid
        - The two GRU layers read the sequence of inputs from left to right.
        - A dense plus sigmoid layer makes a prediction for  𝑦⟨𝑡⟩.
        - Because  𝑦 is a binary value (0 or 1), we use a sigmoid output at the last layer to estimate the chance of the output being 1, corresponding to the user having just said "activate".

        Unidirectional RNN
        - Note that we use a unidirectional RNN rather than a bidirectional RNN.
        - This is because we consider that focusing on past context is more critical than future context for detecting the activation word. This simplification reduces computational overhead, as the model does not process information in both directions.
        - In a real application, trigger word detection could be performed every second using a sliding 10-second window. Therefore, even if the activation word could appear at any point within these 10 seconds, it can be detected immediately upon being spoken. If the algorithm initially fails to detect it, further detections with each subsequent second provide additional opportunities for recognition. This approach efficiently balances performance with timely detection attempts.                

        Function creating the model's graph in Keras.
        
        Argument:
        input_shape -- shape of the model's input data (using Keras conventions)

        Returns:
        model -- Keras model instance
        """
        self._model = Sequential([
            Input(shape = (self._Tx, self._n_freq)),
            Conv1D(196, 15, strides=(4,)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.8),

            GRU(128, return_sequences=True),
            Dropout(0.8),
            BatchNormalization(),

            GRU(128, return_sequences=True),
            Dropout(0.8),
            BatchNormalization(),
            Dropout(0.8),

            TimeDistributed(Dense(1, activation="sigmoid"))
        ])
        self._model.summary()
        #model = Model(inputs = X_input, outputs = X)

    def LoadModel(self):
        """
        Load a pre-trained model which was trained for about 3 hours on a GPU using the architecture defined in BuildModel(), and a large training set of about 4000 examples.
        """
        with open('./models/trigger_word_detection.json', 'r') as f:
            loaded_model_json = f.read()
            self._model = model_from_json(loaded_model_json)
            self._model.load_weights('./models/trigger_word_detection.h5')
        if self._model:
            self._model.summary()
            # If you are going to fine-tune a pretrained model, it is important that you block the weights of all your batchnormalization layers.
            self._model.layers[2].trainable = False
            self._model.layers[7].trainable = False
            self._model.layers[10].trainable = False

    def TrainEvaluateModel(self):
        self._model.compile(loss=BinaryCrossentropy, optimizer=Adam(learning_rate=self._learning_rate, beta_1 = self._beta_1, beta_2 = self._beta2), metrics=["accuracy"])
        history = self._model.fit(self._X, self._Y, batch_size = self._batch_size, epochs=self._epochs)
        PlotModelHistory("Trigger Word Detection", history)
        loss, acc, = self._model.evaluate(self._X_dev, self._Y_dev)
        print(f"Dev set accuracy = {acc}")

    def detect_triggerword(self, filename):
        plt.subplot(2, 1, 1)
        
        # Correct the amplitude of the input file before prediction 
        audio_clip = AudioSegment.from_wav(filename)
        audio_clip = self._match_target_amplitude(audio_clip, -20.0)
        file_handle = audio_clip.export("tmp.wav", format="wav")
        filename = "tmp.wav"

        x = self._graph_spectrogram(filename)
        # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
        x  = x.swapaxes(0,1)
        x = numpy.expand_dims(x, axis=0)
        predictions = self._model.predict(x)
        
        plt.subplot(2, 1, 2)
        plt.plot(predictions[0,:,0])
        plt.ylabel('probability')
        plt.show()
        return predictions

    def chime_on_activate(self, filename, predictions, threshold, output):
        """
        Once you've estimated the probability of having detected the word "activate" at each output step, you can trigger a "chiming" sound to play when the probability is above a certain threshold.
        𝑦⟨𝑡⟩ might be near 1 for many values in a row after "activate" is said, yet we want to chime only once.
        So we will insert a chime sound at most once every 75 output steps.
        This will help prevent us from inserting two chimes for a single instance of "activate".
        This plays a role similar to non-max suppression from computer vision.        
        """
        audio_clip = AudioSegment.from_wav(filename)
        chime = AudioSegment.from_wav(self._chime_file)
        Ty = predictions.shape[1]
        # Step 1: Initialize the number of consecutive output steps to 0
        consecutive_timesteps = 0
        i = 0
        # Step 2: Loop over the output steps in the y
        while i < Ty:
            # Step 3: Increment consecutive output steps
            consecutive_timesteps += 1
            # Step 4: If prediction is higher than the threshold for 20 consecutive output steps have passed
            if consecutive_timesteps > 20:
                # Step 5: Superpose audio and background using pydub
                audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
                # Step 6: Reset consecutive output steps to 0
                consecutive_timesteps = 0
                i = 75 * (i // 75 + 1)
                continue
            # if amplitude is smaller than the threshold reset the consecutive_timesteps counter
            if predictions[0, i, 0] < threshold:
                consecutive_timesteps = 0
            i += 1
        if output:
            audio_clip.export(output, format='wav')

    # Preprocess the audio to the correct format
    def preprocess_audio(self, filename):
        """
        If your audio recording is not 10 seconds, this function will either trim or pad it as needed to make it 10 seconds.
        """
        # Trim or pad audio segment to 10000ms
        padding = AudioSegment.silent(duration=10000)
        segment = AudioSegment.from_wav(filename)[:10000]
        segment = padding.overlay(segment)
        # Set frame rate to 44100
        segment = segment.set_frame_rate(44100)
        # Export as wav
        segment.export(filename, format='wav')

    # Used to standardize volume of audio clip
    def _match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)
    
    # Load a wav file
    def _get_wav_info(self, wav_file):
        rate, data = wavfile.read(wav_file)
        return rate, data
    
    # Calculate and plot spectrogram for a wav audio file
    def _graph_spectrogram(self, wav_file):
        rate, data = self._get_wav_info(wav_file)
        nfft = 200 # Length of each window segment
        fs = 8000 # Sampling frequencies
        noverlap = 120 # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, noverlap = noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = plt.specgram(data[:,0], NFFT=nfft, Fs=fs, noverlap = noverlap)
        return pxx
    
    def _insert_audio_clip(self, background, audio_clip, previous_segments):
        """
        Insert a new audio segment over the background noise at a random time step, ensuring that the 
        audio segment does not overlap with existing segments.
        
        Arguments:
        background -- a 10 second background audio recording.  
        audio_clip -- the audio clip to be inserted/overlaid. 
        previous_segments -- times where audio segments have already been placed
        
        Returns:
        new_background -- the updated background audio
        """
        # Get the duration of the audio clip in ms
        segment_ms = len(audio_clip)
        
        # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
        # the new audio clip. (≈ 1 line)
        segment_time = self.get_random_time_segment(segment_ms)
        
        # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
        # picking new segment_time at random until it doesn't overlap. To avoid an endless loop
        # we retry 5 times(≈ 2 lines)
        retry = 5 
        while self.IsOverlapping(segment_time, previous_segments) and retry >= 0:
            segment_time = self.get_random_time_segment(segment_ms)
            retry = retry - 1
            #print(segment_time)
        # if last try is not overlaping, insert it to the background
        if not self.IsOverlapping(segment_time, previous_segments):
            # Step 3: Append the new segment_time to the list of previous_segments (≈ 1 line)
            previous_segments.append(segment_time)
            # Step 4: Superpose audio segment and background
            new_background = background.overlay(audio_clip, position = segment_time[0])
        else:
            #print("Timeouted")
            new_background = background
            segment_time = (10000, 10000)
        return new_background, segment_time
    
    def _load_raw_audio(self):
        self._activates = []
        self._backgrounds = []
        self._negatives = []
        for filename in os.listdir(f"{self._path}/activates"):
            if filename.endswith("wav"):
                activate = AudioSegment.from_wav(f"{self._path}/activates/{filename}")
                self._activates.append(activate)
        for filename in os.listdir(f"{self._path}/backgrounds"):
            if filename.endswith("wav"):
                background = AudioSegment.from_wav(f"{self._path}/backgrounds/{filename}")
                self._backgrounds.append(background)
        for filename in os.listdir(f"{self._path}/negatives"):
            if filename.endswith("wav"):
                negative = AudioSegment.from_wav(f"{self._path}/negatives/{filename}")
                self._negatives.append(negative)
    
def IsOverlapping_test():
    print(f"=== {IsOverlapping_test.__name__} ===")
    Tx = 5511 # The number of time steps input to the model from the spectrogram
    n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
    Ty = 1375 # The number of time steps in the output of our model
    nsamples = 32
    trigger = TrigerWordDetection("data/raw_data", None, Tx, Ty, n_freq, nsamples, 1e-6, 0.9, 0.999, 16, 5)
    assert trigger.IsOverlapping((670, 1430), []) == False, "Overlap with an empty list must be False"
    assert trigger.IsOverlapping((500, 1000), [(100, 499), (1001, 1100)]) == False, "Almost overlap, but still False"
    assert trigger.IsOverlapping((750, 900), [(100, 750), (1001, 1100)]) == True, "Must overlap with the end of first segment"
    assert trigger.IsOverlapping((750, 1250), [(300, 600), (1250, 1500)]) == True, "Must overlap with the beginning of second segment"
    assert trigger.IsOverlapping((750, 1250), [(300, 600), (600, 1500), (1600, 1800)]) == True, "Is contained in second segment"
    assert trigger.IsOverlapping((800, 1100), [(300, 600), (900, 1000), (1600, 1800)]) == True, "New segment contains the second segment"
    assert not trigger.IsOverlapping((950, 1430), [(2000, 2550), (260, 949)])
    assert trigger.IsOverlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
    print("\033[92m All tests passed!")

def insert_audio_clip_test():
    print(f"=== {insert_audio_clip_test.__name__} ===")
    Tx = 5511 # The number of time steps input to the model from the spectrogram
    n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
    Ty = 1375 # The number of time steps in the output of our model
    nsamples = 32
    trigger = TrigerWordDetection("data/raw_data", None, Tx, Ty, n_freq, nsamples, 1e-6, 0.9, 0.999, 16, 5)

    audio_clip, segment_time = trigger.InsertAudioClip(0, 0, [(0, 4400)])
    duration = segment_time[1] - segment_time[0]
    #print(f"xx: {segment_time}")
    assert audio_clip
    assert segment_time[0] > 4400, "Error: The audio clip is overlapping with the first segment"
    #assert segment_time == (7286, 8201), f"Wrong segment. Expected: (7286, 8201) got:{segment_time}"
    
    # Not possible to insert clip into background
    audio_clip, segment_time = trigger.InsertAudioClip(0, 0, [(0, 9999)])
    assert segment_time == (10000, 10000), "Segment must match the out by max-retry mark"
    print("\033[92m All tests passed!")

def insert_ones_test():
    print(f"=== {insert_ones_test.__name__} ===")
    Tx = 5511 # The number of time steps input to the model from the spectrogram
    n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
    Ty = 1375 # The number of time steps in the output of our model
    nsamples = 32
    trigger = TrigerWordDetection("data/raw_data", None, Tx, Ty, n_freq, nsamples, 1e-6, 0.9, 0.999, 16, 5)

    segment_end_y = random.randrange(0, Ty - 50) 
    segment_end_ms = int(segment_end_y * 10000.4) / Ty;    
    arr1 = trigger.insert_ones(numpy.zeros((1, Ty)), segment_end_ms)

    assert type(arr1) == numpy.ndarray, "Wrong type. Output must be a numpy array"
    assert arr1.shape == (1, Ty), "Wrong shape. It must match the input shape"
    assert numpy.sum(arr1) == 50, "It must insert exactly 50 ones"
    assert arr1[0][segment_end_y - 1] == 0, f"Array at {segment_end_y - 1} must be 0"
    assert arr1[0][segment_end_y] == 0, f"Array at {segment_end_y} must be 0"
    assert arr1[0][segment_end_y + 1] == 1, f"Array at {segment_end_y + 1} must be 1"
    assert arr1[0][segment_end_y + 50] == 1, f"Array at {segment_end_y + 50} must be 1"
    assert arr1[0][segment_end_y + 51] == 0, f"Array at {segment_end_y + 51} must be 0"
    
    arr1 = trigger.insert_ones(numpy.zeros((1, Ty)), 9632)
    assert numpy.sum(arr1) == 50, f"Expected sum of 50, but got {numpy.sum(arr1)}"
    arr1 = trigger.insert_ones(numpy.zeros((1, Ty)), 9637)
    assert numpy.sum(arr1) == 49, f"Expected sum of 49, but got {numpy.sum(arr1)}"
    arr1 = trigger.insert_ones(numpy.zeros((1, Ty)), 10008)
    assert numpy.sum(arr1) == 0, f"Expected sum of 0, but got {numpy.sum(arr1)}"
    arr1 = trigger.insert_ones(numpy.zeros((1, Ty)), 10000)
    assert numpy.sum(arr1) == 0, f"Expected sum of 0, but got {numpy.sum(arr1)}"
    arr1 = trigger.insert_ones(numpy.zeros((1, Ty)), 9996)
    assert numpy.sum(arr1) == 0, f"Expected sum of 0, but got {numpy.sum(arr1)}"
    arr1 = trigger.insert_ones(numpy.zeros((1, Ty)), 9990)
    assert numpy.sum(arr1) == 1, f"Expected sum of 1, but got {numpy.sum(arr1)}"
    arr1 = trigger.insert_ones(numpy.zeros((1, Ty)), 9980)
    assert numpy.sum(arr1) == 2, f"Expected sum of 2, but got {numpy.sum(arr1)}"

    print("\033[92m All tests passed!")

def create_training_example_test():
    print(f"=== {create_training_example_test.__name__} ===")
    Tx = 5511 # The number of time steps input to the model from the spectrogram
    n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
    Ty = 1375 # The number of time steps in the output of our model
    nsamples = 32
    trigger = TrigerWordDetection("data/raw_data", None, Tx, Ty, n_freq, nsamples, 1e-6, 0.9, 0.999, 16, 5)

    numpy.random.seed(18)
    x, y = trigger.create_training_example(0)
    
    assert type(x) == numpy.ndarray, "Wrong type for x"
    assert type(y) == numpy.ndarray, "Wrong type for y"
    assert tuple(x.shape) == (101, 5511), "Wrong shape for x"
    assert tuple(y.shape) == (1, 1375), "Wrong shape for y"
    assert numpy.all(x > 0), "All x values must be higher than 0"
    assert numpy.all(y >= 0), "All y values must be higher or equal than 0"
    assert numpy.all(y <= 1), "All y values must be smaller or equal than 1"
    assert numpy.sum(y) >= 50, "It must contain at least one activate"
    assert numpy.sum(y) % 50 == 0, "Sum of activate marks must be a multiple of 50"
    #assert numpy.isclose(numpy.linalg.norm(x), 39745552.52075), f"Spectrogram is wrong. Check the parameters passed to the insert_audio_clip function. Got {numpy.linalg.norm(x)}"

    print("\033[92m All tests passed!")

def model_tests():
    print(f"=== {model_tests.__name__} ===")
    Tx = 5511 # The number of time steps input to the model from the spectrogram
    n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
    Ty = 1375 # The number of time steps in the output of our model
    nsamples = 32
    trigger = TrigerWordDetection("data/raw_data", "data/XY_dev", Tx, Ty, n_freq, nsamples, 1e-6, 0.9, 0.999, 16, 5) # X_dev.npy: 106.17 MB, X.npy is 135.89 MB
    trigger.BuildModel()
    trigger.LoadModel()
    trigger.TrainEvaluateModel()
    filename = "data/raw_data/dev/1.wav"
    prediction = trigger.detect_triggerword(filename)
    trigger.chime_on_activate(filename, prediction, 0.5, "output/1_output.wav")
    filename = "data/raw_data/dev/2.wav"
    prediction = trigger.detect_triggerword(filename)
    trigger.chime_on_activate(filename, prediction, 0.5, "output/2_output.wav")

if __name__ == "__main__":
    IsOverlapping_test()
    insert_audio_clip_test()
    insert_ones_test()
    create_training_example_test()
    model_tests()