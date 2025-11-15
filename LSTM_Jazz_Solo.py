import numpy, copy
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest
from music21 import *
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from utils.GPU import InitializeGPU
from utils.TrainingMetricsPlot import PlotModelHistory

from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

class LSTM_Jazz_Solo():
    _path:str = None
    _chords = None
    _abstract_grammars = None
    _corpus = None
    _Tx: int = None
    _X = None
    _Y = None
    _N_tones:int = None
    _N_values: int = None
    _indices_tones = None
    _hidden_dim: int = None
    _a0 = None
    _c0 = None
    _learning_rate:float = None
    _model: Model = None
    def __init__(self, path:str, tx:int, hidden_dim:int, values:int, learning_rate:float):
        self._path = path
        self._Tx = tx
        self._hidden_dim = hidden_dim
        self._N_values = values
        self._learning_rate = learning_rate
        self._PrepareData()

    def BuildTrainModel(self, epochs:float):
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
        self._reshaper = Reshape((1, self._N_values))                  # Used in Step 2.B of djmodel(), below
        self._LSTM_cell = LSTM(self._hidden_dim, return_state = True)         # Used in Step 2.C
        self._densor = Dense(self._N_values, activation='softmax')     # Used in Step 2.D
        # Get the shape of input values
        n_values = self._densor.units
        print(f"n_values: {n_values}")
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
            print(f"x: {x.shape}")
            # Step 2.B: Use reshaper to reshape x to be (1, n_values) (≈1 line)
            x = self._reshaper(x)
            #assert (1, n_values) == x.shape
            print(f"x reshaped: {x.shape}")
            # Step 2.C: Perform one step of the LSTM_cell
            _, a, c = self._LSTM_cell(x, initial_state=[a,c])
            # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
            out = self._densor(a)
            # Step 2.E: append the output to "outputs"
            outputs.append(out)
            print(f"out: {out.shape}")
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
        self._model.compile(optimizer=Adam(learning_rate=self._learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        m = 60
        self._a0 = numpy.zeros((m, n_a))
        self._c0 = numpy.zeros((m, n_a))
        history = self._model.fit([self._X, self._a0, self._c0], list(self._Y), epochs=epochs, verbose = 0)
        PlotModelHistory("LSTM Jazz Solo", history)

    def _PrepareData(self):
        """
        What are musical "values"?
        You can informally think of each "value" as a note, which comprises a pitch and duration. For example, if you press down a specific piano key for 0.5 seconds, then you have just played a note. In music theory, a "value" is actually more complicated than this -- specifically, it also captures the information needed to play multiple notes at the same time. For example, when playing a music piece, you might press down two piano keys at the same time (playing multiple notes at the same time generates what's called a "chord"). But you don't need to worry about the details of music theory for this assignment. 

        Music as a sequence of values
        * This python module will obtain a dataset of values, and will use an RNN model to generate sequences of values.
        * This music generation system will use 90 unique values. 
        """
        self._chords, self._abstract_grammars = self._get_musical_data()
        self._corpus, tones, self._tones_indices, self._indices_tones = self._get_corpus_data()
        self._X, self._Y, self._N_tones = self._data_processing(self._tones_indices, 60, 30)   
        print('number of training examples:', self._X.shape[0])
        print('Tx (length of sequence):', self._X.shape[1])
        print('total # of unique values:', self._N_tones)
        print('shape of X:', self._X.shape)
        print('Shape of Y:', self._Y.shape)
        print('Number of chords', len(self._chords))

    ''' Get musical data from a MIDI file '''
    def _get_musical_data(self):
        print(f"\n=== {self._get_musical_data.__name__} ===")
        measures, chords = self._separate_parts_from_midi() #__parse_midi(data_fn)
        abstract_grammars = self.__get_abstract_grammars(measures, chords)
        print(f"{len(measures)} measures, {len(chords)} chords, {len(abstract_grammars)} abstract_grammars")
        return chords, abstract_grammars

    ''' Get corpus data from grammatical data '''
    def _get_corpus_data(self):
        corpus = [x for sublist in self._abstract_grammars for x in sublist.split(' ')]
        values = set(corpus)
        val_indices = dict((v, i) for i, v in enumerate(values))
        indices_val = dict((i, v) for i, v in enumerate(values))
        return corpus, values, val_indices, indices_val

    def _data_processing(self, values_indices, m = 60, Tx = 30):
        # cut the corpus into semi-redundant sequences of Tx values
        Tx = Tx 
        N_values = len(set(self._corpus))
        X = numpy.zeros((m, Tx, N_values), dtype=numpy.bool)
        Y = numpy.zeros((m, Tx, N_values), dtype=numpy.bool)
        for i in range(m):
    #         for t in range(1, Tx):
            random_idx = rng.choice(len(self._corpus) - Tx)
            corp_data = self._corpus[random_idx:(random_idx + Tx)]
            for j in range(Tx):
                idx = values_indices[corp_data[j]]
                if j != 0:
                    X[i, j, idx] = 1
                    Y[i, j-1, idx] = 1
        Y = numpy.swapaxes(Y,0,1)
        Y = Y.tolist()
        return numpy.asarray(X), numpy.asarray(Y), N_values

    def __get_abstract_grammars(self, measures, chords):
        # extract grammars
        abstract_grammars = []
        for ix in range(1, len(measures)):
            m = stream.Voice()
            for i in measures[ix]:
                m.insert(i.offset, i)
            c = stream.Voice()
            for j in chords[ix]:
                c.insert(j.offset, j)
            parsed = self._parse_melody(m, c)
            abstract_grammars.append(parsed)
        return abstract_grammars
    
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

    def _separate_parts_from_midi(self):
        """
        Parses a MIDI file and separates it into individual parts.
        Prints instrument names and returns the list of parts.
        """
        #try:
        # Parse the MIDI file into a score stream
        score = converter.parse(self._path)

        # Access individual parts
        parts = score.parts
        
        if not parts:
            print("The MIDI file might be Type 0 (single track) or has no distinct parts.")
            # If it's a single track, you might need to use more complex methods 
            # to algorithmically separate melody from accompaniment within that track.
            return [score] # Return the whole score as a single part list

        print(f"Found {len(parts)} parts in the MIDI file:")

        separated_parts = []
        for i, part in enumerate(parts):
            # Get the instrument name for identification
            inst = part.getInstrument()
            print(f"Part {i+1}: {inst.instrumentName}")
            separated_parts.append(part)
            
            # Example: Assign the first part as melody, the rest as accompaniment
            if i == 0:
                melody_part = part
            else:
                # You might need to combine other parts into an accompaniment stream
                pass

        # You can now work with individual part streams, for example:
        if melody_part:
            melody_part.show('text')
            accompaniment_stream = stream.Stream(separated_parts[1:]) # Combine others
            # Full stream containing both the melody and the accompaniment. 
            # All parts are flattened. 
            full_stream = stream.Voice()
            full_stream.append(accompaniment_stream)
            full_stream.append(melody_part)
            # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
            # Note that for different instruments (with stream.flat), you NEED to use
            # stream.Part(), not stream.Voice().
            # Accompanied solo is in range [478, 548)
            solo_stream = stream.Voice()
            for part in full_stream:
                print(f"\npart: {part}")
                #part.show("text")
                curr_part = stream.Part()
                if isinstance(part, stream.Part):
                    print(f"\ninstrument.Instrument:")
                    for i in part.getElementsByClass(instrument.Piano):
                        print(f"instrument: {i}")
                        i.show("text")
                        curr_part.append(i)
                    print(f"\ntempo.MetronomeMark:")
                    for i in part.getElementsByClass(tempo.MetronomeMark):
                        print(f"MetronomeMark: {i}")
                        i.show("text")
                        curr_part.append(i)
                    print(f"\nkey.KeySignature:")
                    for i in part.getElementsByClass(key.KeySignature):
                        print(f"KeySignature: {i}")
                        i.show("text")
                        curr_part.append(i)
                    print(f"\nmeter.TimeSignature:")
                    for i in part.getElementsByClass(meter.TimeSignature):
                        print(f"TimeSignature: {i}")
                        i.show("text")
                        curr_part.append(i)
                    print(f"\nmeter.getElementsByOffset:")
                    for i in part.getElementsByOffset(476, 548, includeEndBoundary=True):
                        print(f"offset: {i}")
                        i.show("text")
                        curr_part.append(i)
                    cp = curr_part.flatten()
                    solo_stream.insert(cp)

            # Group by measure so you can classify. 
            # Note that measure 0 is for the time signature, metronome, etc. which have
            # an offset of 0.0.
            melody_stream = solo_stream[-1]
            measures = OrderedDict()
            offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]
            measureNum = 0 # for now, don't use real m. nums (119, 120)
            for key_x, group in groupby(offsetTuples, lambda x: x[0]):
                measures[measureNum] = [n[1] for n in group]
                measureNum += 1

            # Get the stream of chords.
            # offsetTuples_chords: group chords by measure number.
            chordStream = solo_stream[0]
            chordStream.removeByClass(note.Rest)
            chordStream.removeByClass(note.Note)
            offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

            # Generate the chord structure. Use just track 1 (piano) since it is
            # the only instrument that has chords. 
            # Group into 4s, just like before. 
            chords = OrderedDict()
            measureNum = 0
            for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
                chords[measureNum] = [n[1] for n in group]
                measureNum += 1

            # Fix for the below problem.
            #   1) Find out why len(measures) != len(chords).
            #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
            #           actually show up, while the accompaniment's beat 1 right after does.
            #           Actually on second thought: melody/comp start on Ab, and resolve to
            #           the same key (Ab) so could actually just cut out last measure to loop.
            #           Decided: just cut out the last measure. 
            #del chords[len(chords) - 1]
            print(f"{len(chords)} chords, {len(measures)} measures")
            assert len(chords) == len(measures)
            return measures, chords
        return None, None

if __name__ == "__main__":
    jazz = LSTM_Jazz_Solo("data/original_metheny.mid", 30, 64, 90, 0.01)
    jazz.BuildTrainModel(300)