import numpy, copy
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest
from music21 import *
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

# https://github.com/evancchow/jazzml
# https://github.com/jisungk/deepjazz
def __parse_midi_details(data_fn):
    print(f"\n=== {__parse_midi_details.__name__} ===")
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(data_fn)
    assert len(midi_data.getElementsByClass(stream.Part)) == len(midi_data.parts)
    allnotes = []
    allchords = []
    melody_voice = {}
    idxPart = 0
    timeSig = []
    mmark = []
    for part in midi_data.recurse().getElementsByClass(stream.Part):
        print(f"\npart {idxPart}: {part}")
        part.show("text")
        idxPart += 1
        idxMeasure = 0
        for measure in part.recurse().getElementsByClass(stream.Measure):
            print(f"measure {idxMeasure}: {measure}")
            measure.show("text")
            idxMeasure += 1
            timeSig.append(measure.getElementsByClass(meter.TimeSignature))
            mmark.append(measure.getElementsByClass(tempo.MetronomeMark))
            idxVoice = 0
            for voice in measure.recurse().getElementsByClass(stream.Voice):
                print(f"voice {idxVoice}: {voice.offset}, {voice}")
                voice.show("text")
                idxVoice += 1
                if voice.offset not in melody_voice:
                    melody_voice[voice.offset] = [voice]
                else:
                    melody_voice[voice.offset].append(voice)
                notes = voice.recurse().getElementsByClass(note.Note).notes
                chords = voice.recurse().getElementsByClass(chord.Chord)
                #for i in notes:
                allnotes.append(notes)
                #for i in chords:
                allchords.append(chords)
            """
            for voice in measure.getElementsByClass(stream.Voice):
                print(f"voice {idxVoice}: {voice.offset}, {voice}")
                voice.show("text")
                idxVoice += 1
                if voice.offset not in melody_voice:
                    melody_voice.add(voice.offset, voice)
                notes = voice.getElementsByClass(note.Note).notes
                chords = voice.getElementsByClass(chord.Chord)
                for i in notes:
                    allnotes.append(i)
                for i in chords:
                    allchords.append(i)
            """
    #print(f"\nmmark:")
    #mmark.stream().show("text")
    #print(f"\ntimeSig:")
    #timeSig.stream().show("text")
    print(f"\n{len(timeSig)} timeSig, {len(mmark)} mmark")
    print(f"\n{len(allchords)} chords, {len(allnotes)} notes, {len(melody_voice)} melody_voice") # 8075 chords, 15390 notes, 9766 melody_voice
    print(f"\nmelody_voice: {melody_voice}")

def __parse_midi_part(data_fn, partIndex:int = None):
    print(f"\n=== {__parse_midi_part.__name__} ===")
    # Parse the MIDI data for separate melody and accompaniment parts.
    # Part-Measure-Voice.
    # melody_stream = midi_data[5]     # For Metheny piece, Melody is Part #5.
    midi_data = converter.parse(data_fn)
    assert len(midi_data.getElementsByClass(stream.Part)) == len(midi_data.parts)
    allnotes = []
    allchords = []
    melody_voice = []
    idxPart = 0
    timeSig: stream.Part = None
    mmark: stream.Part = None
    idxPart += 1
    idxMeasure = 0
    part = midi_data.getElementsByClass(stream.Part)[partIndex]
    print(f"\npart {partIndex}:")
    part.show("text")
    for measure in part.getElementsByClass(stream.Measure):
        print(f"measure {idxMeasure}: {measure}")
        measure.show("text")
        idxMeasure += 1
        tmp = measure.getElementsByClass(meter.TimeSignature)
        if tmp:
            timeSig = tmp
        tmp = measure.getElementsByClass(tempo.MetronomeMark)
        if tmp:
            mmark = tmp
        idxVoice = 0
        for voice in measure.getElementsByClass(stream.Voice()):
            print(f"voice {idxVoice}: {voice.offset}, {voice}")
            voice.show("text")
            idxVoice += 1
            melody_voice.insert(voice.offset, voice)
            notes = voice.getElementsByClass(note.Note).notes
            chords = voice.getElementsByClass(chord.Chord)
            for i in notes:
                allnotes.append(i)
            for i in chords:
                allchords.append(i)
    print(f"\nmmark:")
    mmark.stream().show("text")
    print(f"\ntimeSig:")
    timeSig.stream().show("text")
    print(f"\n{len(allchords)} chords, {len(allnotes)} notes, {len(melody_voice)} melody_voice") # 82 chords, 232 notes, 84 melody_voice
    #print(f"MetronomeMark: {mmark.number}, TimeSignature: {timeSig.numerator / timeSig.denominator}")
    print("\nFullName, CommonName, Len, Offset")
    for i in allchords:
        print(f"{i.fullName}, {i.pitchedCommonName}, {i.quarterLength}, {float(i.offset)}")

    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25
    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    # Also add Electric Guitar. 
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))
    print(f"{len(melody_voice)} melody_voice")

    # The accompaniment parts. Take only the best subset of parts from
    # the original data. Maybe add more parts, hand-add valid instruments.
    # Should at least add a string part (for sparse solos).
    # Verified are good parts: 0, 1, 6, 7 '''
    partIndices = [0, 1, 6, 7]
    accompaniment = stream.Voice()
    #print("\nenumerate(midi_data)")
    #for i, j in enumerate(midi_data):
    #    if i in partIndices and not isinstance(j, metadata.Metadata):
    #        print(f"{i}: {j}, {j}")
    #        j.show("text")

    #comp_stream = measure_stream.voices.stream()
    accompaniment.append([j.flatten() for i, j in enumerate(midi_data) if i in partIndices and not isinstance(j, metadata.Metadata)])

    print(f"\naccompaniment: {len(accompaniment)}")
    #comp_stream.show("text")

    # Full stream containing both the melody and the accompaniment. 
    # All parts are flattened. 
    full_stream = stream.Voice()

    #full_stream = measure_stream.voices.stream()
    for i in accompaniment:
        full_stream.append(i)

    full_stream.append(melody_voice)
    print(f"\nfull_stream: {len(full_stream)}")
    solo_stream = stream.Voice()
    for part in full_stream:
        #print(f"\npart: {part}")
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
    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
    # Note that for different instruments (with stream.flat), you NEED to use
    # stream.Part(), not stream.Voice().
    # Accompanied solo is in range [478, 548)
    #solo_stream = measure_stream.voices.stream()
    print(f"\n{len(solo_stream)} solo_stream")
    solo_stream.show("text")

    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    melody_stream = solo_stream[-1]
    print(f"\n{len(melody_stream)} melody_stream: {melody_stream}")
    melody_stream.show("text")
    measures = OrderedDict()
    # Check data/original_metheny.log. All Measures are 4 distance away from one another, i.e., {0.0}, {4.0}, {8.0}, ...
    offsetTuples = [(n.offset // 4, n) for n in melody_stream]
    measureNum = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1
    print(f"\nmeasures: {len(measures)} {measures}")

    solo_stream[0].show("text")
    chordStream = solo_stream[0] # Only piano has chord
    chordStream.removeByClass(instrument.Piano)
    chordStream.removeByClass(tempo.MetronomeMark)
    chordStream.removeByClass(key.Key)   
    chordStream.removeByClass(meter.TimeSignature)
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    print(f"\n{len(chordStream)} chordStream: {chordStream}")
    chordStream.show("text")
    # Check data/original_metheny.log. All Measures are 4 distance away from one another, i.e., {0.0}, {4.0}, {8.0}, ...
    offsetTuples_chords = [(n.offset // 4, n) for n in chordStream]
    #offsetTuples_chords = [(n.offset // 4, n) for n in allchords]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1
    print(f"\nchords: {len(chords)} {chords}")
    # Fix for the below problem.
    #   1) Find out why len(measures) != len(chords).
    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
    #           actually show up, while the accompaniment's beat 1 right after does.
    #           Actually on second thought: melody/comp start on Ab, and resolve to
    #           the same key (Ab) so could actually just cut out last measure to loop.
    #           Decided: just cut out the last measure. 
    #del chords[len(chords) - 1]
    assert len(chords) == len(measures), f"{len(chords)} chords, {len(measures)} measures" # AssertionError: 28 chords, 21 measures
    print(f"{len(chords)} measures and chords")
    return measures, chords

''' Helper function to parse a MIDI file into its measures and chords '''
def __parse_midi(data_fn):
    """
    A common arrangement of nested Streams is a Score Stream containing one or more Part Streams, each Part Stream in turn containing one or more Measure Streams.
    Parts > Measures > Voice

    You might think that there should be a convenience property .measures to get all the measures. But the problem with that is that measure numbers would be quite different from index numbers. 
    For instance, most pieces (that don’t have pickup measures) begin with measure 1, not zero. 
    Sometimes there are measure discontinuities within a piece (e.g., some people number first and second endings with the same measure number). 
    For that reason, gathering Measures is best accomplished NOT with getElementsByClass(stream.Measure) but instead with either the measures() method (returning a Stream of Parts or Measures) or the measure() method (returning a single Measure).    
    """
    print(f"\n=== {__parse_midi.__name__} ===")
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(data_fn)
    assert len(midi_data.getElementsByClass(stream.Part)) == len(midi_data.parts)
    #print(f"Parts (getElementsByClass): {len(midi_data.getElementsByClass(stream.Part))}")
    #print(f"\nParts (.parts): {len(midi_data.parts)}")
    measure_part5_len = len(midi_data.getElementsByClass(stream.Part)[5].getElementsByClass(stream.Measure))
    measure_part5 = midi_data.parts[5].measures(1, measure_part5_len)
    measure_stream = midi_data.parts[5].getElementsByClass(stream.Measure)
    print(f"\nMeasure (Part[5]): {len(measure_part5)} {measure_part5}")
    #print(f"\nmeasure_part5.show:")
    #measure_part5.show('text')
    #print("recurse():")
    #for el in midi_data.recurse().getElementsByClass(stream.Voice):
    #    print(el.offset, el, el.activeSite)
    # Get melody part, compress into single voice.
    #melody_stream = midi_data[5]     # For Metheny piece, Melody is Part #5.
    print(f"\nMelody Voice: {len(measure_part5.recurse().getElementsByClass(stream.Voice))}")
    melody_voice = []
    for el in measure_part5.recurse().getElementsByClass(stream.Voice):
        #print(el.offset, el, el.activeSite)
        print(f"el offset: {el.offset}, {el}")
        melody_voice.insert(int(el.offset), el)

    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25

    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    # Also add Electric Guitar. 
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))
    print(f"{len(melody_voice)} melody_voice")

    # The accompaniment parts. Take only the best subset of parts from
    # the original data. Maybe add more parts, hand-add valid instruments.
    # Should at least add a string part (for sparse solos).
    # Verified are good parts: 0, 1, 6, 7 '''
    partIndices = [0, 1, 6, 7]
    accompaniment = stream.Voice()
    #print("\nenumerate(midi_data)")
    #for i, j in enumerate(midi_data):
    #    if i in partIndices and not isinstance(j, metadata.Metadata):
    #        print(f"{i}: {j}, {j}")
    #        j.show("text")

    #comp_stream = measure_stream.voices.stream()
    accompaniment.append([j.flatten() for i, j in enumerate(midi_data) if i in partIndices and not isinstance(j, metadata.Metadata)])

    print(f"\naccompaniment: {len(accompaniment)}")
    #comp_stream.show("text")

    # Full stream containing both the melody and the accompaniment. 
    # All parts are flattened. 
    full_stream = stream.Voice()

    #full_stream = measure_stream.voices.stream()
    for i in range(len(accompaniment)):
        full_stream.append(accompaniment[i])

    full_stream.append(melody_voice)

    print(f"\nfull_stream: {len(full_stream)}")
    #full_stream.show("text")
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
    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
    # Note that for different instruments (with stream.flat), you NEED to use
    # stream.Part(), not stream.Voice().
    # Accompanied solo is in range [478, 548)
    #solo_stream = measure_stream.voices.stream()
    print(f"\nsolo_stream: {len(solo_stream)}")
    solo_stream.show("text")
    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    melody_stream = solo_stream[-1]
    print(f"\n{len(melody_stream)} melody_stream: {melody_stream}")
    melody_stream.show("text")
    measures = OrderedDict()
    # Check data/original_metheny.log. All Measures are 4 distance away from one another, i.e., {0.0}, {4.0}, {8.0}, ...
    offsetTuples = [(n.offset // 4, n) for n in melody_stream]
    measureNum = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1
    print(f"\nmeasures: {len(measures)} {measures}")
    # Get the stream of chords.
    # offsetTuples_chords: group chords by measure number.
    chordStream = solo_stream[0] # Only piano has chord
    chordStream.removeByClass(instrument.Piano)
    chordStream.removeByClass(tempo.MetronomeMark)
    chordStream.removeByClass(key.Key)   
    chordStream.removeByClass(meter.TimeSignature)
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    print(f"\n{len(chordStream)} chordStream: {chordStream}")
    chordStream.show("text")

    # Check data/original_metheny.log. All Measures are 4 distance away from one another, i.e., {0.0}, {4.0}, {8.0}, ...
    offsetTuples_chords = [(n.offset // 4, n) for n in chordStream]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1
    print(f"\nchords: {len(chords)} {chords}")
    # Fix for the below problem.
    #   1) Find out why len(measures) != len(chords).
    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
    #           actually show up, while the accompaniment's beat 1 right after does.
    #           Actually on second thought: melody/comp start on Ab, and resolve to
    #           the same key (Ab) so could actually just cut out last measure to loop.
    #           Decided: just cut out the last measure. 
    #del chords[len(chords) - 1]
    assert len(chords) == len(measures), f"{len(chords)} chords, {len(measures)} measures"
    print(f"{len(chords)} measures and chords")
    return measures, chords

''' Helper function to determine if a note is a scale tone. '''
def __is_scale_tone(chord, note):
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
def __is_approach_tone(chord, note):
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
def __is_chord_tone(lastChord, note):
    return (note.name in (p.name for p in lastChord.pitches))

def parse_melody(fullMeasureNotes, fullMeasureChords):
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
        elif __is_scale_tone(lastChord, nr):
            elementType = 'S'
        # A: Check if it's an approach tone, i.e. +-1 halfstep chord tone.
        elif __is_approach_tone(lastChord, nr):
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

def __get_abstract_grammars(measures, chords):
    # extract grammars
    abstract_grammars = []
    for ix in range(1, len(measures)):
        m = stream.Voice()
        for i in measures[ix]:
            m.insert(i.offset, i)
        c = stream.Voice()
        for j in chords[ix]:
            c.insert(j.offset, j)
        parsed = parse_melody(m, c)
        abstract_grammars.append(parsed)
    return abstract_grammars

''' Get musical data from a MIDI file '''
def get_musical_data(data_fn):
    print(f"\n=== {get_musical_data.__name__} ===")
    measures, chords = __parse_midi(data_fn)
    abstract_grammars = __get_abstract_grammars(measures, chords)
    print(f"{len(measures)} measures, {len(chords)} chords, {len(abstract_grammars)} abstract_grammars")
    return chords, abstract_grammars

''' Get corpus data from grammatical data '''
def get_corpus_data(abstract_grammars):
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))

    return corpus, values, val_indices, indices_val

def data_processing(corpus, values_indices, m = 60, Tx = 30):
    # cut the corpus into semi-redundant sequences of Tx values
    Tx = Tx 
    N_values = len(set(corpus))
    X = numpy.zeros((m, Tx, N_values), dtype=numpy.bool)
    Y = numpy.zeros((m, Tx, N_values), dtype=numpy.bool)
    for i in range(m):
#         for t in range(1, Tx):
        random_idx = rng.choice(len(corpus) - Tx)[0]
        corp_data = corpus[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = values_indices[corp_data[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j-1, idx] = 1
    
    Y = numpy.swapaxes(Y,0,1)
    Y = Y.tolist()
    return numpy.asarray(X), numpy.asarray(Y), N_values 

def load_music_utils(file):
    chords, abstract_grammars = get_musical_data(file)
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones, chords)

def separate_parts(midi_file_path):
    # Parse the MIDI file into a music21 Stream (Score) object
    score = converter.parse(midi_file_path)

    # Use partitionByInstrument() to organize the score into parts based on instrument
    # Note: This might not work perfectly if the original file doesn't have clear instrument definitions per track
    parts = instrument.partitionByInstrument(score)

    if parts:
        # Iterate through the parts and identify them (e.g., as melody or accompaniment)
        for i, part in enumerate(parts):
            print(f"Part {i+1} Instrument: {part.getInstrument().instrumentName}")
            # You can add logic here to determine which part is the melody/accompaniment
            # based on the instrument name (e.g., "Flute", "Acoustic Guitar")
            
            # Example: Accessing notes in each part
            notes_and_chords = part.flatten().notesAndRests
            print(f"  Contains {len(notes_and_chords)} notes/rests.")
            # Do further processing with the part data

        # You can access specific parts by their index (e.g., first part is typically index 0)
        # melody_part = parts[0]
        # accompaniment_part = parts[1] # If you know the structure of your specific MIDI
        return parts

    else:
        # If partitionByInstrument doesn't work, the MIDI file might have a single track
        # or ambiguous instrument data. You would have to apply a custom logic (e.g., 
        # filtering by pitch range or rhythmic complexity) to manually separate notes
        print("Could not automatically partition by instrument. Accessing parts directly.")
        all_parts = []
        for part in score.parts:
            all_parts.append(part)
            print(f"Found part: {part[0].bestName()}") # prints the most likely instrument
        return all_parts
    
if __name__ == "__main__":
    #__parse_midi_details("data/original_metheny.mid")
    #__parse_midi_part("data/original_metheny.mid", 5) # AssertionError: 28 chords, 21 measures
    parts = separate_parts("data/original_metheny.mid")
    # You can then perform analysis or write the parts to separate MIDI files
    # if needed:
    if parts:
        melody = parts[0] # assuming the first part is melody
        melody.write('midi', 'melody.mid') # $ fluidsynth melody.mid
    """
    X, Y, n_values, indices_values, chords = load_music_utils('data/original_metheny.mid') # AssertionError: 28 chords, 21 measures
    print('number of training examples:', X.shape[0])
    print('Tx (length of sequence):', X.shape[1])
    print('total # of unique values:', n_values)
    print('shape of X:', X.shape)
    print('Shape of Y:', Y.shape)
    print('Number of chords', len(chords))
    """
    """
    number of training examples: 60
    Tx (length of sequence): 30
    total # of unique values: 90
    shape of X: (60, 30, 90)
    Shape of Y: (30, 60, 90)
    Number of chords 19    
    """