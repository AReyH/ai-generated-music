import glob
import pickle
from music21 import converter, instrument, note, chord
from pathlib import Path


def load_songs(path):
    # Loads MIDIs and converts them into a list of sequence of notes
    songs = []
    folder = Path(path)
    for file in folder.rglob('*.mid'):
        songs.append(file)

    return songs

def preprocess_songs(songs)    
    notes = []
    for i,file in enumerate(songs):
        print(f'{i+1}: {file}')
        try:
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except:
            print(f'FAILED: {i+1}: {file}')


    # Save notes to Drive for future usage
    with open('notes', 'wb') as filepath:
        pickle.dump(notes, filepath)


if __name__ == '__main__':
    path = '/songs/'
    songs = load_songs(path)
    preprocess_songs(songs)