import argparse, numpy, copy, traceback, tensorflow as tf
from itertools import zip_longest
from pathlib import Path
from music21 import *
from keras import saving
from tensorflow.keras.saving import serialize_keras_object
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Layer, RepeatVector
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotModelHistory
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

@saving.register_keras_serializable()
class ArgMaxLayer(Layer):
    def call(self, x):
        return tf.math.argmax(x, axis=-1)
@saving.register_keras_serializable()
class OneHotLayer(Layer):
    _n_values: int = None
    def __init__(self, n_values:int, **kwargs):
        super().__init__(**kwargs)
        self._n_values = n_values
    def call(self, x):
        return tf.one_hot(x, self._n_values)
    def get_config(self):
        """
        get_config(): This method should return a dictionary containing all the arguments needed to reconstruct an instance of your class.
        """
        base_config = super().get_config()
        config = {
            "n_values": serialize_keras_object(self._n_values),
            # Serialize any other custom objects or non-base type 
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        """
        from_config(): This is a class method that takes a configuration dictionary and returns a new instance of your class.
        """
        n_values_config = config.pop("n_values")
        n_values = tf.keras.saving.deserialize_keras_object(n_values_config)
        return cls(n_values, **config)
    
class LSTM_Jazz_Solo():
    _path:str = None
    _chords = None
    _abstract_grammars = None
    _corpus = None
    _Tx: int = None # length of the sequences in the corpus. Tx LSTM cells where each cell is responsible for learning the following note based on the previous note and context.
    _Ty: int = None # #time steps to generate
    _X = None
    _Y = None
    _batch_size:int = None
    _N_tones:int = None
    _N_values: int = None
    _indices_tones = None
    _hidden_dim: int = None
    _a0 = None
    _c0 = None
    _learning_rate:float = None
    _model: Model = None
    _inference_model: Model = None
    _model_path:str = None
    _inference_model_path:str = None
    def __init__(self, path:str, model_path, inference_model_path:str, tx:int, hidden_dim:int, values:int, learning_rate:float, batch_size:int):
        """
        tx: length of the sequences in the corpus. Tx LSTM cells where each cell is responsible for learning the following note based on the previous note and context.
        ty: #time steps to generate
        """
        self._path = path
        self._model_path = model_path
        self._inference_model_path = inference_model_path
        self._Tx = tx
        self._hidden_dim = hidden_dim
        self._N_values = values
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._PrepareData()
        self._reshaper = Reshape((1, self._N_values))                  # Used in Step 2.B of djmodel(), below
        self._LSTM_cell = LSTM(self._hidden_dim, return_state = True)         # Used in Step 2.C
        self._densor = Dense(self._N_values, activation='softmax')     # Used in Step 2.D
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            self._model = load_model(self._model_path)
        if self._inference_model_path and len(self._inference_model_path) and Path(self._inference_model_path).exists() and Path(self._inference_model_path).is_file():
            print(f"Using saved model {self._inference_model_path}...")
            self._inference_model = load_model(self._inference_model_path)

    def BuildTrainModel(self, epochs:int, verbose:int = 0, retrain: bool = False):
        """
        #### Sequence generation uses a for-loop
        * If you're building an RNN where, at test time, the entire input sequence x1, x2, ..., x(T) is given in advance, then Keras has simple built-in functions to build the model. 
        * However, for **sequence generation, at test time you won't know all the values of x(t) in advance**.
        * Instead, you'll generate them one at a time using x(t) = y(t-1). 
            * The input at time "t" is the prediction at the previous time step "t-1".
        * So you'll need to implement your own for-loop to iterate over the time steps. 

        #### Shareable weights
        * The function `djmodel()` will call the LSTM layer $T_x$ times using a for-loop.
        * It is important that all $T_x$ copies have the same weights. 
            - The $T_x$ steps should have shared weights that aren't re-initialized.
        * Referencing a globally defined shared layer will utilize the same layer-object instance at each time step.
        * The key steps for implementing layers with shareable weights in Keras are: 
        1. Define the layer objects (you'll use global variables for this).
        2. Call these objects when propagating the input.

        #### 3 types of layers
        * The layer objects you need for global variables have been defined.  
            - [Reshape()](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape): Reshapes an output to a certain shape.
            - [LSTM()](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM): Long Short-Term Memory layer
            - [Dense()](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense): A regular fully-connected neural network layer.

        Implement the djmodel composed of Tx LSTM cells where each cell is responsible
        for learning the following note based on the previous note and context.
        Each cell has the following schema: 
                [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()
        Arguments:
            Tx -- length of the sequences in the corpus
            LSTM_cell -- LSTM layer instance
            densor -- Dense layer instance
            reshaper -- Reshape layer instance
        
        Returns:
            model -- a keras instance model with inputs [X, a0, c0]
        """
        print(f"\n=== {self.BuildTrainModel.__name__} ===")
        new_model = not self._model
        if not self._model:
            # Get the shape of input values
            n_values = self._densor.units
            # Get the number of the hidden state vector
            n_a = self._LSTM_cell.units
            
            # Define the input layer and specify the shape
            X = Input(shape=(self._Tx, n_values)) 
            
            # Define the initial hidden state a0 and initial cell state c0
            # using `Input`
            a0 = Input(shape=(n_a,), name='a0')
            c0 = Input(shape=(n_a,), name='c0')
            a = a0
            c = c0
            # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
            outputs = []
            
            # Step 2: Loop over tx
            for t in range(self._Tx):
                # Step 2.A: select the "t"th time step vector from X. 
                # X has the shape (m, Tx, n_values).
                # The shape of the 't' selection should be (n_values,).
                x = X[:,t,:]
                # Step 2.B: Use reshaper to reshape x to be (1, n_values) (≈1 line)
                x = self._reshaper(x)
                #assert (1, n_values) == x.shape
                # Step 2.C: Perform one step of the LSTM_cell
                _, a, c = self._LSTM_cell(x, initial_state=[a,c])
                # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
                out = self._densor(a)
                # Step 2.E: append the output to "outputs"
                outputs.append(out)
            # Step 3: Create model instance
            self._model = Model(inputs=[X, a0, c0], outputs=outputs)
            self._model.summary()
            plot_model(
                self._model,
                to_file=f"output/LSTM_Jazz_Solo.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
            self._model.compile(optimizer=Adam(learning_rate=self._learning_rate, beta_1=0.9, beta_2=0.999, weight_decay=0.01), loss=CategoricalCrossentropy(), metrics=["accuracy"] * len(outputs))
        if new_model or retrain:
            self._a0 = numpy.zeros((self._batch_size, n_a))
            self._c0 = numpy.zeros((self._batch_size, n_a))
            history = self._model.fit([self._X, self._a0, self._c0], list(self._Y), epochs=epochs, verbose = verbose)
            PlotModelHistory("LSTM Jazz Solo", history)
            if self._model_path:
                self._model.save(self._model_path)
                print(f"Model saved to {self._model_path}.")

    def BuildInferenceModel(self, ty: int, retrain: bool = False):
        """
        Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
        
        Arguments:
        LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
        densor -- the trained "densor" from model(), Keras layer object
        Ty -- integer, number of time steps to generate
        
        Returns:
        inference_model -- Keras model instance
        """
        print(f"\n=== {self.BuildInferenceModel.__name__} ===")
        new_model = not self._inference_model
        if not self._inference_model:
            self._Ty = ty
            # Get the shape of input values
            n_values = self._densor.units
            # Get the number of the hidden state vector
            n_a = self._LSTM_cell.units
            
            # Define the input of your model with a shape 
            x0 = Input(shape=(1, n_values))
            
            # Define s0, initial hidden state for the decoder LSTM
            a0 = Input(shape=(n_a,), name='a0')
            c0 = Input(shape=(n_a,), name='c0')
            a = a0
            c = c0
            x = x0

            # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
            outputs = []
            
            # Step 2: Loop over Ty and generate a value at every time step
            for t in range(self._Ty):
                # Step 2.A: Perform one step of LSTM_cell. Use "x", not "x0" (≈1 line)
                _, a, c = self._LSTM_cell(x, initial_state=[a,c])
                
                # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                out = self._densor(a)
                # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 90) (≈1 line)
                outputs.append(out)
        
                # Step 2.D: 
                # Select the next value according to "out",
                # Set "x" to be the one-hot representation of the selected value
                # See instructions above.
                #print(f"out: {out.shape}")
                #x = tf.math.argmax(out, axis=-1)
                x = ArgMaxLayer()(out)
                #print(f"argmax: {x}")
                #x = tf.one_hot(x, n_values)
                x = OneHotLayer(n_values)(x)
                # Step 2.E: 
                # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
                x = RepeatVector(1)(x)
                
            # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
            self._inference_model = Model(inputs=[x0,a0,c0], outputs=outputs)
            self._inference_model.summary()
            plot_model(
                self._inference_model,
                to_file=f"output/LSTM_Jazz_Solo_inference.png",
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                show_layer_activations=True)
            if self._inference_model_path:
                self._inference_model.save(self._inference_model_path)
                print(f"Model saved to {self._inference_model_path}.")

    def predict_and_sample(self):
        """
        Predicts the next value of values using the inference model.
        
        Arguments:
        x_initializer -- numpy array of shape (1, 1, 90), one-hot vector initializing the values generation
        a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
        c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
        
        Returns:
        results -- numpy-array of shape (Ty, 90), matrix of one-hot vectors representing the values generated
        indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
        """
        x_initializer = numpy.zeros((1, 1, self._N_values))
        a_initializer = numpy.zeros((1, self._LSTM_cell.units))
        c_initializer = numpy.zeros((1, self._LSTM_cell.units))        
        n_values = x_initializer.shape[2]
        
        # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
        pred = self._inference_model.predict([x_initializer, a_initializer, c_initializer])
        # Step 2: Convert "pred" into an numpy.array() of indices with the maximum probabilities
        indices = tf.math.argmax(pred, axis=-1)
        # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
        results = to_categorical(indices, num_classes=n_values)
        return results, indices

    def generate_music(self, path:str, diversity = 0.5):
        """
        Generates music using a model trained to learn musical patterns of a jazz soloist. Creates an audio stream
        to save the music and play it.
        
        Arguments:
        model -- Keras model Instance, output of djmodel()
        indices_tones -- a python dictionary mapping indices (0-77) into their corresponding unique tone (ex: A,0.250,< m2,P-4 >)
        temperature -- scalar value, defines how conservative/creative the model is when generating music
        
        Returns:
        predicted_tones -- python list containing predicted tones
        """
        print(f"\n=== {self.generate_music.__name__} ===")
        # set up audio stream
        out_stream = stream.Stream()
        
        # Initialize chord variables
        curr_offset = 0.0                                     # variable used to write sounds to the Stream.
        num_chords = int(len(self._music_data['chords']) / 3)                     # number of different set of chords
        
        print("Predicting new values for different set of chords.")
        # Loop over all 18 set of chords. At each iteration generate a sequence of tones
        # and use the current chords to convert it into actual sounds 
        for i in range(1, num_chords):
            
            # Retrieve current chord from stream
            curr_chords = stream.Voice()
            
            # Loop over the chords of the current set of chords
            #print(f"chords: {type(self._music_data['chords'])} {self._music_data['chords'][:10]}")
            for j in self._music_data['chords'][i]:
                # Add chord to the current chords with the adequate offset, no need to understand this
                curr_chords.insert((j.offset % 4), j)
                #print(type(j))
            
            # Generate a sequence of tones using the model
            _, indices = self.predict_and_sample()
            indices = indices.numpy()
            print(f"indices: {type(indices)} {indices.shape}")
            indices = list(indices.squeeze())
            print(f"{len(indices)} indices: {type(indices[0])}")
            print(f"self._indices_tones: {type(self._indices_tones)}")
            pred = [self._indices_tones[p] for p in indices]
            
            predicted_tones = 'C,0.25 '
            for k in range(len(pred) - 1):
                predicted_tones += pred[k].pitchedCommonName + ' ' 
            
            predicted_tones +=  pred[-1].pitchedCommonName
                    
            #### POST PROCESSING OF THE PREDICTED TONES ####
            # We will consider "A" and "X" as "C" tones. It is a common choice.
            predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

            # Pruning #1: smoothing measure
            predicted_tones = self._prune_grammar(predicted_tones)
            
            # Use predicted tones and current chords to generate sounds
            sounds = self._unparse_grammar(predicted_tones, curr_chords)

            # Pruning #2: removing repeated and too close together sounds
            sounds = self._prune_notes(sounds)

            # Quality assurance: clean up sounds
            sounds = self._clean_up_notes(sounds)

            # Print number of tones/notes in sounds
            print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
            
            # Insert sounds into the output stream
            for m in sounds:
                out_stream.insert(curr_offset + m.offset, m)
            for mc in curr_chords:
                out_stream.insert(curr_offset + mc.offset, mc)

            curr_offset += 4.0
            
        # Initialize tempo of the output stream with 130 bit per minute
        out_stream.insert(0.0, tempo.MetronomeMark(number=130))

        # Save audio stream to fine
        mf = midi.translate.streamToMidiFile(out_stream)
        mf.open(path, 'wb')
        mf.write()
        print(f"Your generated music is saved in {path}")
        mf.close()
        
        # Play the final stream through output (see 'play' lambda function above)
        # play = lambda x: midi.realtime.StreamPlayer(x).play()
        # play(out_stream)
        return out_stream
    
    ''' Given a grammar string and chords for a measure, returns measure notes. '''
    def _unparse_grammar(self, m1_grammar, m1_chords):
        m1_elements = stream.Voice()
        currOffset = 0.0 # for recalculate last chord.
        prevElement = None
        for ix, grammarElement in enumerate(m1_grammar.split(' ')):
            terms = grammarElement.split(',')
            if len(terms) > 1:
                currOffset += float(terms[1]) # works just fine

            # Case 1: it's a rest. Just append
            if terms[0] == 'R':
                rNote = note.Rest(quarterLength = float(terms[1]))
                m1_elements.insert(currOffset, rNote)
                continue

            # Get the last chord first so you can find chord note, scale note, etc.
            try: 
                lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]
            except IndexError:
                m1_chords[0].offset = 0.0
                lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]

            # Case: no < > (should just be the first note) so generate from range
            # of lowest chord note to highest chord note (if not a chord note, else
            # just generate one of the actual chord notes). 

            # Case #1: if no < > to indicate next note range. Usually this lack of < >
            # is for the first note (no precedent), or for rests.
            if (len(terms) == 2): # Case 1: if no < >.
                insertNote = note.Note() # default is C

                # Case C: chord note.
                if terms[0] == 'C':
                    insertNote = self.__generate_chord_tone(lastChord)

                # Case S: scale note.
                elif terms[0] == 'S':
                    insertNote = self.__generate_scale_tone(lastChord)

                # Case A: approach note.
                # Handle both A and X notes here for now.
                else:
                    insertNote = self.__generate_approach_tone(lastChord)

                # Update the stream of generated notes
                insertNote.quarterLength = float(terms[1])
                print(f"insertNote: {insertNote}")
                if not insertNote.octave or insertNote.octave < 4:
                    insertNote.octave = 4
                m1_elements.insert(currOffset, insertNote)
                prevElement = insertNote

            # Case #2: if < > for the increment. Usually for notes after the first one.
            else:
                # Get lower, upper intervals and notes.
                interval1 = interval.Interval(terms[2].replace("<",''))
                interval2 = interval.Interval(terms[3].replace(">",''))
                if interval1.cents > interval2.cents:
                    upperInterval, lowerInterval = interval1, interval2
                else:
                    upperInterval, lowerInterval = interval2, interval1
                lowPitch = interval.transposePitch(prevElement.pitch, lowerInterval)
                highPitch = interval.transposePitch(prevElement.pitch, upperInterval)
                numNotes = int(highPitch.ps - lowPitch.ps + 1) # for range(s, e)

                # Case C: chord note, must be within increment (terms[2]).
                # First, transpose note with lowerInterval to get note that is
                # the lower bound. Then iterate over, and find valid notes. Then
                # choose randomly from those.
                
                if terms[0] == 'C':
                    relevantChordTones = []
                    for i in range(0, numNotes):
                        currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                        if self. __is_chord_tone(lastChord, currNote):
                            relevantChordTones.append(currNote)
                    if len(relevantChordTones) > 1:
                        insertNote = rng.choice([i for i in relevantChordTones
                            if i.nameWithOctave != prevElement.nameWithOctave])
                    elif len(relevantChordTones) == 1:
                        insertNote = relevantChordTones[0]
                    else: # if no choices, set to prev element +-1 whole step
                        insertNote = prevElement.transpose(rng.choice([-2,2]))
                    if not insertNote.octave or insertNote.octave < 3:
                        insertNote.octave = 3
                    insertNote.quarterLength = float(terms[1])
                    m1_elements.insert(currOffset, insertNote)

                # Case S: scale note, must be within increment.
                elif terms[0] == 'S':
                    relevantScaleTones = []
                    for i in range(0, numNotes):
                        currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                        if self. __is_scale_tone(lastChord, currNote):
                            relevantScaleTones.append(currNote)
                    if len(relevantScaleTones) > 1:
                        insertNote = rng.choice([i for i in relevantScaleTones
                            if i.nameWithOctave != prevElement.nameWithOctave])
                    elif len(relevantScaleTones) == 1:
                        insertNote = relevantScaleTones[0]
                    else: # if no choices, set to prev element +-1 whole step
                        insertNote = prevElement.transpose(rng.choice([-2,2]))
                    if not insertNote.octave or insertNote.octave < 3:
                        insertNote.octave = 3
                    insertNote.quarterLength = float(terms[1])
                    m1_elements.insert(currOffset, insertNote)

                # Case A: approach tone, must be within increment.
                # For now: handle both A and X cases.
                else:
                    relevantApproachTones = []
                    for i in range(0, numNotes):
                        currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                        if self. __is_approach_tone(lastChord, currNote):
                            relevantApproachTones.append(currNote)
                    if len(relevantApproachTones) > 1:
                        insertNote = rng.choice([i for i in relevantApproachTones
                            if i.nameWithOctave != prevElement.nameWithOctave])
                    elif len(relevantApproachTones) == 1:
                        insertNote = relevantApproachTones[0]
                    else: # if no choices, set to prev element +-1 whole step
                        insertNote = prevElement.transpose(rng.choice([-2,2]))
                    if not insertNote.octave or insertNote.octave < 3:
                        insertNote.octave = 3
                    insertNote.quarterLength = float(terms[1])
                    m1_elements.insert(currOffset, insertNote)

                # update the previous element.
                prevElement = insertNote
        return m1_elements
    
    ''' Helper function to generate a chord tone. '''
    def __generate_chord_tone(self, lastChord):
        lastChordNoteNames = [p.nameWithOctave for p in lastChord.pitches]
        return note.Note(rng.choice(lastChordNoteNames))

    ''' Helper function to generate a scale tone. '''
    def __generate_scale_tone(self, lastChord):
        # Derive major or minor scales (minor if 'other') based on the quality
        # of the lastChord.
        scaleType = scale.WeightedHexatonicBlues() # minor pentatonic
        if lastChord.quality == 'major':
            scaleType = scale.MajorScale()
        # Can change later to deriveAll() for flexibility. If so then use list
        # comprehension of form [x for a in b for x in a].
        scales = scaleType.derive(lastChord) # use deriveAll() later for flexibility
        allPitches = list(set([pitch for pitch in scales.getPitches()]))
        allNoteNames = [i.name for i in allPitches] # octaves don't matter

        # Return a note (no octave here) in a scale that matches the lastChord.
        sNoteName = rng.choice(allNoteNames)
        lastChordSort = lastChord.sortAscending()
        sNoteOctave = rng.choice([i.octave for i in lastChordSort.pitches])
        sNote = note.Note(("%s%s" % (sNoteName, sNoteOctave)))
        return sNote

    ''' Helper function to generate an approach tone. '''
    def __generate_approach_tone(self, lastChord):
        sNote = self.__generate_scale_tone(lastChord)
        aNote = sNote.transpose(rng.choice([1, -1]))
        return aNote
    
    def _PrepareData(self):
        """
        What are musical "values"?
        You can informally think of each "value" as a note, which comprises a pitch and duration. For example, if you press down a specific piano key for 0.5 seconds, then you have just played a note. In music theory, a "value" is actually more complicated than this -- specifically, it also captures the information needed to play multiple notes at the same time. 
        For example, when playing a music piece, you might press down two piano keys at the same time (playing multiple notes at the same time generates what's called a "chord"). But you don't need to worry about the details of music theory for this assignment. 

        Music as a sequence of values
        * This python module will obtain a dataset of values, and will use an RNN model to generate sequences of values.
        * This music generation system will use 90 unique values. 
        """
        self._extract_complex_grammar1()
        self._corpus = self._music_data['chords']
        self._tones_indices, self._indices_tones = self._get_corpus_data()
        self._X, self._Y, self._N_tones = self._data_processing(30)
        if isinstance(self._music_data, dict):
            print(f"Total Measures: {len(self._music_data['measures'])}")
            print(f"Unique Grammars Found: {self._music_data['unique_grammar_count']}")
            print("-" * 30)
            print("Sample Vocabulary (First 15 Unique Tokens):")
            # Print a sample of the complex names found
            for chord_name, token in list(self._music_data['vocabulary_map'].items())[:15]:
                print(f"{token} : {chord_name}")
            print("-" * 30)
            print(f"Grammar Sequence Length: {len(self._music_data['grammar_sequence'])}")
            print("First 20 tokens of the sequence:")
            print(self._music_data['grammar_sequence'][:20])
        else:
            print(self._music_data)        
        print(f'number of training examples: {self._X.shape[0]}')
        print(f'Tx (length of sequence): {self._X.shape[1]}')
        print(f'total # of unique values: {self._N_tones}')
        print(f'shape of X: {self._X.shape}')
        print(f'Shape of Y: {self._Y.shape}')
        print(f"# chords: {len(self._music_data['chords'])}, type: {type(self._music_data['chords'][0])}")
        print(f"# chord symbols: {len(self._music_data['chord_symbols'])}, type: {type(self._music_data['chord_symbols'][0])}")

    def _extract_complex_grammar(self):
        """
        With the follup-up question to Gemini, it gives me 2 choices and this is choice-A.

        The reason you are only getting ~26 unique grammars is likely because the previous code looked only at the Guitar (Melody) track.
        In MIDI files, melody tracks are often monophonic (one note at a time) or sparse. To capture the harmonic complexity of Pat Metheny's music (extended chords like maj7, #11, 13, etc.) and reach a vocabulary size of ~90, you must analyze the accompaniment (Piano and Bass) or the entire score simultaneously.
        Here is the updated solution using music21. This version uses chordify() to squash all instruments into a single timeline, creating a full harmonic analysis for every measure. It also uses Pitch Class Sets (Normal Order) instead of simple names to distinguish complex jazz voicings (e.g., distinguishing a Cmaj7 from a Cmaj9).

        Key Changes Made
        score.chordify(): Instead of isolating the guitar part, this function collapses the Piano, Bass, Strings, and Guitar into a single stream. This ensures that if the Bass plays a 'C' and the Piano plays 'E G B', the code recognizes a C Major 7 chord, rather than just isolated notes.
        measure_chord.normalOrder: Instead of looking for generic names like "C Major" (which groups many variations together), this looks at the exact Pitch Class Set (e.g., (0, 4, 7, 10)). This is far more granular and will significantly increase the number of unique grammars found, likely hitting your target of ~90.    
        """
        print(f"\n=== {self._extract_complex_grammar.__name__} ===")
        # 1. Load the MIDI file
        try:
            score = converter.parse(self._path)
        except Exception as e:
            return f"Error loading file: {e}"

        # 2. 'Chordify' the Score
        # This compresses all tracks (Guitar, Piano, Bass, Strings) into one 
        # vertical harmonic structure. This captures the full chord voicing.
        print("Chordifying score (analyzing all instruments)...")
        chordified_score = score.chordify()

        extracted_chords = []
        extracted_measures = []
        extracted_grammars = []
        
        # 3. Iterate through measures
        # We use the chordified score to ensure we capture the harmony of the measure
        measures = chordified_score.makeMeasures()
        
        for m in measures.getElementsByClass('Measure'):
            extracted_measures.append(m.number)
            
            # Flatten the measure to get all simultaneous notes played by all instruments
            notes = m.flatten().notes
            
            if len(notes) > 0:
                # Create a chord object from all notes in this measure
                measure_chord = chord.Chord(notes)
                
                # 4. Generate Abstract Grammar Token
                # To get a high count (90+), we use 'normalOrder'.
                # This is a tuple of integers representing the Pitch Classes.
                # Example: C Major = (0, 4, 7). C Major 7 = (0, 4, 7, 11).
                # This distinguishes specific harmonic colors (7ths, 9ths, etc.)
                grammar_token = tuple(measure_chord.normalOrder)
                extracted_grammars.append(grammar_token)
                extracted_chords.append(measure_chord)
            #else:
            #    extracted_grammars.append("Rest")

        # 5. Create Abstract Grammar Mapping
        # Identify unique chord structures
        #unique_vocabulary = sorted(list(set(extracted_grammars)), key=lambda x: str(x))
        unique_vocabulary = list(set(extracted_chords))
        
        # Map each unique pitch set to an ID (0, 1, 2...)
        grammar_map = {token: i for i, token in enumerate(unique_vocabulary)}
        
        # Convert sequence to IDs
        grammar_sequence = [grammar_map[g] for g in extracted_chords]
        assert self._N_values == len(unique_vocabulary), f"Unique grammar count mismatch. Expects: {self._N_values}, gets: {len(unique_vocabulary)}"
        self._music_data = {
            "measures": extracted_measures,
            "chords": extracted_chords,
            "chord_symbols": extracted_grammars,
            "unique_grammar_count": len(unique_vocabulary),
            "unique_vocabulary": unique_vocabulary,
            "vocabulary_map": grammar_map,
            "grammar_sequence": grammar_sequence
        }

    def _extract_complex_grammar1(self):
        """
        To increase the number of unique abstract grammars from 26 to roughly 90, we must change how the chords are identified.
        The previous code reduced chords to their simplest triads (Root + Major/Minor). Since this is a Pat Metheny track ("And Then I Knew"), the music relies on Jazz Harmony (7ths, 9ths, 11ths, suspensions, and slash chords). "C Major", "C Major 7", and "C Major 9" were previously lumped into a single "C Major" token.
        Here is the updated code. It uses music21.harmony to analyze chord extensions and bass notes, which will drastically increase the vocabulary size.

        Key Changes Made
        Multi-Track Aggregation: The code now looks for 'Piano', 'String', 'Pad', and 'Organ' tracks in addition to 'Guitar'. Identifying chords from a single melody line (monophonic) is mathematically impossible to yield 90 chords; the harmony comes from the interaction of the instruments.
        Harmonic Analysis (harmony.chordSymbolFigureFromChord): Instead of simple root/quality, we now use Music21's jazz harmony analyzer.
        Old result: C-major
        New result: Cmaj7, C7b9, Csus4, C/E (Slash chords).
        Pitched Common Name Fallback: If the chord is too complex for a standard symbol (common in fusion jazz), it falls back to names like "f-minor seventh chord" or "dominant ninth chord", ensuring distinct tokens for distinct sounds.
        This approach should significantly widen the vocabulary to meet your expectation of ~90 unique grammars.

        The perceived difference in vocabulary format (tuples of integers vs. tuples of strings) likely stems from the analytic step immediately following chord identification, which is often called Abstract Pitch Class Set (APCS) analysis or Grammatical Rule extraction.Here is a breakdown of the differences and a recommendation for which one to use.
        1. Why the Difference in Vocabulary Format? The difference lies not just in the data type, but in the level of abstraction required for the analysis.
        Feature	                Choice A (Simple Grammar)	                    Choice B (Complex Grammar)
        Analysis Focus	        Triads and Functional Harmony (Major/Minor).	Extended Harmony and Jazz Voicings (7ths, 9ths, alterations).
        Primary Vocabulary	    Strings (e.g., 'C-major').	                    Strings (e.g., 'Cmaj7#11').
        Why Tuples of Integers?	This often occurs if the string chords (e.g., 'C-major') are converted to their Abstract Pitch Class Sets (APCS). C-Major → {0,4,7}. This set is then standardized, possibly resulting in an integer tuple like (0,3,7) (the standard minor trichord, if transposed).	
        Why Tuples of Strings?	This often occurs when analyzing transitional grammar or Markov Chains. The "vocabulary" here represents the rules or states of movement, e.g., (’Cmaj7’,’Fmaj7’), representing the move from the I to the IV.

        In short, the two versions of the function I provided both produce a vocabulary of chord name strings, which are then typically converted to integers for the sequence.
        If you are seeing tuples:
        Tuples of Integers relate to pitch content analysis (like APCS).
        Tuples of Strings relate to sequence transition analysis (like a Markov model).

        2. Which One Is the Right One to Use? The "right" choice depends entirely on your analytic goal.Since your source material is a piece by Pat Metheny, a master of Jazz Fusion, the harmonic detail captured by Choice B is vastly superior and more appropriate for a meaningful analysis of his unique style.
        Choice	    Best Used For...	                                                                Why?
        Choice A	Tonal Function Analysis (I→V→vi), or analyzing simple Folk/Pop music.	            It ignores all the color tones (7ths, 9ths, etc.), focusing only on the basic major/minor scaffolding. This loses the core sound of the piece.
        Choice B	Extended Harmonic Analysis and Style Characterization (Jazz, Fusion, Contemporary).	It captures the specific harmonic quality (e.g., Cmaj7#11 vs. Cmaj7) that defines Pat Metheny's complex voice-leading and chord vocabulary. This high level of detail is necessary to achieve the expected ∼90 unique grammars.    
        
        Conclusion: For analyzing Pat Metheny's music, Choice B (Complex Grammar) is the correct choice, as it preserves the rich harmonic extensions that define the genre.
        """
        print(f"\n=== {self._extract_complex_grammar1.__name__} ===")
        try:
            score = converter.parse(self._path)
            # 1. TIME SIGNATURE FIX (Necessary for makeMeasures stability)
            ts = score.getTimeSignatures()
            if not ts:
                score.insert(0, meter.TimeSignature('4/4'))
            
            # 2. MANUALLY QUANTIZE THE SCORE (Fixes division-by-zero)
            score_flat = score.flatten()
            for element in score_flat.notesAndRests:
                element.offset = round(element.offset * 2.0) / 2.0
                element.duration.quarterLength = round(element.duration.quarterLength * 2.0) / 2.0

            # 3. Aggregate and Analyze
            analyzed_stream = score_flat 
            extracted_chords = []
            extracted_chord_symbols = []
            extracted_measures = []
            measure_stream = analyzed_stream.makeMeasures()

            for m in measure_stream.getElementsByClass('Measure'):
                extracted_measures.append(m.number)
                notes_and_chords = m.flatten().getElementsByClass(['Note', 'Chord'])
                
                # CRITICAL FIX: Use Pitch Class (0-11) for deterministic analysis
                unique_pitch_classes = set()
                for element in notes_and_chords:
                    if isinstance(element, note.Note):
                        unique_pitch_classes.add(element.pitch.pitchClass)
                    elif isinstance(element, chord.Chord):
                        for p in element.pitches:
                            unique_pitch_classes.add(p.pitchClass)

                if len(unique_pitch_classes) > 0:
                    # Enforce Canonical Order: Sort the pitch classes (e.g., {0, 4, 7})
                    canonical_pitch_set = sorted(list(unique_pitch_classes))

                    # Create the aggregate chord from the deterministic pitch classes
                    aggregate_chord = chord.Chord(canonical_pitch_set) 
                    
                    # Use complex jazz analysis
                    try:
                        symbol = harmony.chordSymbolFigureFromChord(aggregate_chord, includeChordType=True)
                        if symbol == 'Chord Symbol Cannot Be Identified' or 'Unnamed Triad' in symbol:
                            symbol = aggregate_chord.pitchedCommonName
                    except:
                        symbol = aggregate_chord.pitchedCommonName
                    extracted_chord_symbols.append(symbol)
                    extracted_chords.append(aggregate_chord)
                #else:
                #    extracted_chords.append("Rest")
            # --- GRAMMAR GENERATION ---
            #print(f"{len(extracted_chords)} extracted_chords: {extracted_chords}")
            #unique_vocabulary = sorted(list(set(extracted_chords))) This list consists of str and tuple. Sorting it will hit an error wit this mixed types.
            unique_vocabulary = list(set(extracted_chords))
            grammar_map = {chord_name: i for i, chord_name in enumerate(unique_vocabulary)}
            grammar_sequence = [grammar_map[c] for c in extracted_chords]
            self._music_data = {
                "measures": extracted_measures,
                "chords": extracted_chords,
                "chord_symbols": extracted_chord_symbols,
                "vocabulary_map": grammar_map,
                "grammar_sequence": grammar_sequence,
                "unique_grammar_count": len(unique_vocabulary),
                "unique_vocabulary": unique_vocabulary
            }
        except Exception as e:
            print(f"Exception: {e}")
            print(traceback.format_exc())

    ''' Get corpus data from grammatical data '''
    def _get_corpus_data(self):
        val_indices = dict((v, i) for i, v in enumerate(self._music_data['vocabulary_map']))
        indices_val = dict((i, v) for i, v in enumerate(self._music_data['vocabulary_map']))
        return val_indices, indices_val

    def _data_processing(self, Tx = 30):
        # cut the corpus into semi-redundant sequences of Tx values
        Tx = Tx 
        N_values = len(set(self._corpus))
        X = numpy.zeros((self._batch_size, Tx, N_values), dtype=numpy.bool)
        Y = numpy.zeros((self._batch_size, Tx, N_values), dtype=numpy.bool)
        for i in range(self._batch_size):
    #         for t in range(1, Tx):
            random_idx = rng.choice(len(self._corpus) - Tx)
            corp_data = self._corpus[random_idx:(random_idx + Tx)]
            for j in range(Tx):
                idx = self._tones_indices[corp_data[j]]
                if j != 0:
                    X[i, j, idx] = 1
                    Y[i, j-1, idx] = 1
        Y = numpy.swapaxes(Y,0,1)
        Y = Y.tolist()
        return numpy.asarray(X), numpy.asarray(Y), N_values

    def _parse_melody(self, fullMeasureNotes, fullMeasureChords):
        # Remove extraneous elements.x
        measure = copy.deepcopy(fullMeasureNotes)
        chords = copy.deepcopy(fullMeasureChords)
        measure.removeByNotOfClass([note.Note, note.Rest])
        chords.removeByNotOfClass([chord.Chord])

        # Information for the start of the measure.
        # 1) measureStartTime: the offset for measure's start, e.g. 476.0.
        # 2) measureStartOffset: how long from the measure start to the first element.
        measureStartTime = measure[0].offset - (measure[0].offset % 4)
        measureStartOffset  = measure[0].offset - measureStartTime

        # Iterate over the notes and rests in measure, finding the grammar for each
        # note in the measure and adding an abstract grammatical string for it. 

        fullGrammar = ""
        prevNote = None # Store previous note. Need for interval.
        numNonRests = 0 # Number of non-rest elements. Need for updating prevNote.
        for ix, nr in enumerate(measure):
            # Get the last chord. If no last chord, then (assuming chords is of length
            # >0) shift first chord in chords to the beginning of the measure.
            try: 
                lastChord = [n for n in chords if n.offset <= nr.offset][-1]
            except IndexError:
                chords[0].offset = measureStartTime
                lastChord = [n for n in chords if n.offset <= nr.offset][-1]

            # FIRST, get type of note, e.g. R for Rest, C for Chord, etc.
            # Dealing with solo notes here. If unexpected chord: still call 'C'.
            elementType = ' '
            # R: First, check if it's a rest. Clearly a rest --> only one possibility.
            if isinstance(nr, note.Rest):
                elementType = 'R'
            # C: Next, check to see if note pitch is in the last chord.
            elif nr.name in lastChord.pitchNames or isinstance(nr, chord.Chord):
                elementType = 'C'
            # L: (Complement tone) Skip this for now.
            # S: Check if it's a scale tone.
            elif self.__is_scale_tone(lastChord, nr):
                elementType = 'S'
            # A: Check if it's an approach tone, i.e. +-1 halfstep chord tone.
            elif self.__is_approach_tone(lastChord, nr):
                elementType = 'A'
            # X: Otherwise, it's an arbitrary tone. Generate random note.
            else:
                elementType = 'X'

            # SECOND, get the length for each element. e.g. 8th note = R8, but
            # to simplify things you'll use the direct num, e.g. R,0.125
            if (ix == (len(measure)-1)):
                # formula for a in "a - b": start of measure (e.g. 476) + 4
                diff = measureStartTime + 4.0 - nr.offset
            else:
                diff = measure[ix + 1].offset - nr.offset

            # Combine into the note info.
            noteInfo = "%s,%.3f" % (elementType, nr.quarterLength) # back to diff

            # THIRD, get the deltas (max range up, max range down) based on where
            # the previous note was, +- minor 3. Skip rests (don't affect deltas).
            intervalInfo = ""
            if isinstance(nr, note.Note):
                numNonRests += 1
                if numNonRests == 1:
                    prevNote = nr
                else:
                    noteDist = interval.Interval(noteStart=prevNote, noteEnd=nr)
                    noteDistUpper = interval.add([noteDist, "m3"])
                    noteDistLower = interval.subtract([noteDist, "m3"])
                    intervalInfo = ",<%s,%s>" % (noteDistUpper.directedName, 
                        noteDistLower.directedName)
                    # print "Upper, lower: %s, %s" % (noteDistUpper,
                    #     noteDistLower)
                    # print "Upper, lower dnames: %s, %s" % (
                    #     noteDistUpper.directedName,
                    #     noteDistLower.directedName)
                    # print "The interval: %s" % (intervalInfo)
                    prevNote = nr

            # Return. Do lazy evaluation for real-time performance.
            grammarTerm = noteInfo + intervalInfo 
            fullGrammar += (grammarTerm + " ")
        return fullGrammar.rstrip()

    ''' Helper function to determine if a note is a scale tone. '''
    def __is_scale_tone(self, chord, note):
        # Method: generate all scales that have the chord notes th check if note is
        # in names

        # Derive major or minor scales (minor if 'other') based on the quality
        # of the chord.
        scaleType = scale.DorianScale() # i.e. minor pentatonic
        if chord.quality == 'major':
            scaleType = scale.MajorScale()
        # Can change later to deriveAll() for flexibility. If so then use list
        # comprehension of form [x for a in b for x in a].
        scales = scaleType.derive(chord) # use deriveAll() later for flexibility
        allPitches = list(set([pitch for pitch in scales.getPitches()]))
        allNoteNames = [i.name for i in allPitches] # octaves don't matter

        # Get note name. Return true if in the list of note names.
        noteName = note.name
        return (noteName in allNoteNames)

    ''' Helper function to determine if a note is an approach tone. '''
    def __is_approach_tone(self, chord, note):
        # Method: see if note is +/- 1 a chord tone.

        for chordPitch in chord.pitches:
            stepUp = chordPitch.transpose(1)
            stepDown = chordPitch.transpose(-1)
            if (note.name == stepDown.name or 
                note.name == stepDown.getEnharmonic().name or
                note.name == stepUp.name or
                note.name == stepUp.getEnharmonic().name):
                    return True
        return False

    ''' Helper function to determine if a note is a chord tone. '''
    def __is_chord_tone(self, lastChord, note):
        return (note.name in (p.name for p in lastChord.pitches))

    ''' Helper function to down num to the nearest multiple of mult. '''
    def __roundDown(self, num, mult):
        return (float(num) - (float(num) % mult))

    ''' Helper function to round up num to nearest multiple of mult. '''
    def __roundUp(self, num, mult):
        return self.__roundDown(num, mult) + mult

    ''' Helper function that, based on if upDown < 0 or upDown >= 0, rounds number 
        down or up respectively to nearest multiple of mult. '''
    def __roundUpDown(self, num, mult, upDown):
        if upDown < 0:
            return self.__roundDown(num, mult)
        else:
            return self.__roundUp(num, mult)

    ''' Helper function, from recipes, to iterate over list in chunks of n 
        length. '''
    def __grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

    ''' Smooth the measure, ensuring that everything is in standard note lengths 
        (e.g., 0.125, 0.250, 0.333 ... ). '''
    def _prune_grammar(self, curr_grammar):
        pruned_grammar = curr_grammar.split(' ')
        for ix, gram in enumerate(pruned_grammar):
            terms = gram.split(',')
            #print(f"terms: {len(terms)} {terms}")
            if len(terms) > 1:
                terms[1] = str(self.__roundUpDown(float(terms[1]), 0.250, rng.choice([-1, 1])))
            pruned_grammar[ix] = ','.join(terms)
        pruned_grammar = ' '.join(pruned_grammar)

        return pruned_grammar

    ''' Remove repeated notes, and notes that are too close together. '''
    def _prune_notes(self, curr_notes):
        for n1, n2 in self.__grouper(curr_notes, n=2):
            if n2 == None: # corner case: odd-length list
                continue
            if isinstance(n1, note.Note) and isinstance(n2, note.Note):
                if n1.nameWithOctave == n2.nameWithOctave:
                    curr_notes.remove(n2)
        return curr_notes

    ''' Perform quality assurance on notes '''
    def _clean_up_notes(self, curr_notes):
        removeIxs = []
        for ix, m in enumerate(curr_notes):
            # QA1: ensure nothing is of 0 quarter note len, if so changes its len
            if (m.quarterLength == 0.0):
                m.quarterLength = 0.250
            # QA2: ensure no two melody notes have same offset, i.e. form a chord.
            # Sorted, so same offset would be consecutive notes.
            if (ix < (len(curr_notes) - 1)):
                if (m.offset == curr_notes[ix + 1].offset and
                    isinstance(curr_notes[ix + 1], note.Note)):
                    removeIxs.append((ix + 1))
        curr_notes = [i for ix, i in enumerate(curr_notes) if ix not in removeIxs]
        return curr_notes
    
if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='LSTM Jazz Solo Music Generation')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    # def self. __init__(self, path:str, tx:int, hidden_dim:int, values:int, learning_rate:float):
    jazz = LSTM_Jazz_Solo("data/original_metheny.mid", "models/LSTM_Jazz_Solo.keras", "models/LSTM_Jazz_Solo_Inference.keras", 30, 64, 211, 0.01, 60)
    jazz.BuildTrainModel(300, args.retrain)
    jazz.BuildInferenceModel(100, args.retrain)
    results, indices = jazz.predict_and_sample()
    print("numpy.argmax(results[12]) =", numpy.argmax(results[12]))
    print("numpy.argmax(results[17]) =", numpy.argmax(results[17]))
    print("list(indices[12:18]) =", list(indices[12:18]))
    jazz.generate_music("output/lstm_jazz.midi")