import numpy, copy
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest
from music21 import *
from utils.TermColour import bcolors
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

#----------------------------HELPER FUNCTIONS----------------------------------#
# https://github.com/cuthbertLab/music21/issues/1813
''' Helper function to parse a MIDI file into its measures and chords '''
def __parse_midi(path:str):
    # Parse the MIDI data for separate melody and accompaniment parts.
    score = converter.parse(path)
    """
    A common arrangement of nested Streams is a Score Stream containing one or more Part Streams, each Part Stream in turn containing one or more Measure Streams.
    Parts > Measures > Voice
    """
    #print(f"Parts: {len(midi_data.getElementsByClass(stream.Part))}")
    #print(f"Parts: {len(midi_data.parts)}")
    #print(f"Measure: {len(midi_data.getElementsByClass(stream.Part)[0].getElementsByClass(stream.Measure))}")
    #print(f"Show: {midi_data.show('text')}")
    melody_stream = score[5]     # For Metheny piece, Melody is Part #5.
    print(f"{melody_stream.show('text')}")
    melody_voice = stream.Voice()
    for part in score.parts:
        # Check part's instrument or name
        print(f"partName: {part.partName}")
        for measure in part.getElementsByClass('Measure'):
            for voice in measure.getElementsByClass('Voice'):
                #print(el.offset, el, el.activeSite)
                if voice.quarterLength == 0.0:
                    voice.quarterLength = 0.25
                melody_voice.insert(voice.offset, voice)
    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    # Also add Electric Guitar. 
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))

    # Get melody part, compress into single voice.
    #print(f"{melody_stream.show('text')}")
    #print(f"Melody Parts: {len(melody_stream.parts)}")
    #print(f"Melody Voice: {len(melody_stream.getElementsByClass(stream.Voice))}")
    #print(f"Melody Voice: {len(melody_stream.getElementsByClass(stream.Part)[0].getElementsByClass(stream.Voice))}")
    #for el in melody_stream.recurse():
    #    print(el.offset, el, el.activeSite)
    melody = melody_voice

    print(f"{len(melody)} melody") # 516 melody
    # The accompaniment parts. Take only the best subset of parts from
    # the original data. Maybe add more parts, hand-add valid instruments.
    # Should add least add a string part (for sparse solos).
    # Verified are good parts: 0, 1, 6, 7 '''
    partIndices = [0, 1, 6, 7]
    comp_stream = stream.Voice()
    #comp_stream.append([j.flat for i, j in enumerate(score)
    #    if i in partIndices])
    comp_stream.append([j.flatten() for i, j in enumerate(score) if i in partIndices and not isinstance(j, metadata.Metadata)])
    print(f"comp_stream: {len(comp_stream)}") # comp_stream: 3
    # Full stream containing both the melody and the accompaniment.
    # All parts are flattened. 
    full_stream = stream.Voice()
    for i in range(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody)

    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
    # Note that for different instruments (with stream.flat), you NEED to use
    # stream.Part(), not stream.Voice().
    # Accompanied solo is in range [478, 548)
    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        if isinstance(part, stream.Part):
            curr_part.append([inst for inst in part.getElementsByClass(instrument.Instrument)])
            curr_part.append([metro for metro in part.getElementsByClass(tempo.MetronomeMark)])
            curr_part.append([key for key in part.getElementsByClass(key.KeySignature)])
            curr_part.append([time for time in part.getElementsByClass(meter.TimeSignature)])
            curr_part.append([e for e in part.getElementsByOffset(476, 548, includeEndBoundary=True)])
            print(f"curr_part: {len(curr_part.flatten())}")
            # curr_part: 142
            # curr_part: 251
            # curr_part: 82
            solo_stream.insert(curr_part.flatten())

    print(f"\n{len(solo_stream)} solo_stream: {solo_stream}") # 3 solo_stream: <music21.stream.Voice 0x7f06cbdc9f40>
    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    melody_stream = solo_stream[-1] # melody was appended last in full_stream in line 63
    print(f"\n{len(melody_stream)} melody_stream: {melody_stream}") # 82 melody_stream: <music21.stream.Part 0x7f06cc5da1d0>
    melody_stream.show("text")
    measures = OrderedDict()
    offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]
    measureNum = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1
    print(f"measureNum: {measureNum}, {len(measures)} measures") # measureNum: 21, 21 measures
    # Get the stream of chords.
    # offsetTuples_chords: group chords by measure number.
    chordStream = solo_stream[0]
    print(f"{len(chordStream)} chordStream") # 142 chordStream
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    print(f"{len(chordStream)} chordStream") # 51 chordStream
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    chordNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[chordNum] = [n[1] for n in group]
        chordNum += 1
    print(f"chordNum: {chordNum}, {len(chords)} chords") # chordNum: 28, 28 chords
    # Fix for the below problem.
    #   1) Find out why len(measures) != len(chords).
    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
    #           actually show up, while the accompaniment's beat 1 right after does.
    #           Actually on second thought: melody/comp start on Ab, and resolve to
    #           the same key (Ab) so could actually just cut out last measure to loop.
    #           Decided: just cut out the last measure. 
    del chords[len(chords) - 1]
    assert len(chords) == len(measures), f"{len(chords)} chords, {len(measures)} measures"
    return measures, chords

if __name__ == "__main__":
    """
    Expected output:
    number of training examples: 60
    Tx (length of sequence): 30
    total # of unique values: 90
    shape of X: (60, 30, 90)
    Shape of Y: (30, 60, 90)
    Number of chords 19    
    """
    measures, chords = __parse_midi("data/original_metheny.mid")  # 'And Then I Knew' by Pat Metheny 