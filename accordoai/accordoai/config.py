
def get_config():
    return {
        "HOP_LENGTH" : 2048,
        "SAMPLE_RATE": 22050,
        "BINS_PER_OCT" : 24,
        "SLICE_SIZE" : 300,
        "MONO": True,
        "VOCAB_PATH" : "resources/vocab.csv",
    }

def get_chords():
    return {
        "roots" : ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N', 'X'],
        "basses" : ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N', 'X'],
        "triads" : ['Major', 'Minor', 'Diminished', 'Augmented', 'Sus2', 'Sus4', 'N', 'X'],
        "fourths" : ['dim7', 'min7', 'maj7', 'maj6', 'N', 'X']
    }
