import numpy as np
import pandas as pd
from config import get_config, get_chords
from scipy import stats
from collections import Counter
from music21 import chord

config = get_config()
slice_size = config['SLICE_SIZE']
hops = config['HOP_LENGTH']
sr = config['SAMPLE_RATE']
chord_vocab_file = config['VOCAB_PATH']

# Chromatic scale in terms of semitone steps from C
chromatic_scale = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic equivalents mapping
ENHARMONIC_FLATS = {
    'A#': 'Bb',
    'D#': 'Eb',
    'G#': 'Ab',
    'B#': 'C',
    'E#': 'F'
}

def postprocess(df, beats, tempo, duration):
    try:
        # Step 1: Get chords with labels
        chords = vector_to_chord(df)

        try:
            # Optional: group chords to beats and trim to duration
            # chords = group_chords(aligned_chords_df=chords, beat_times=beats, tempo=tempo, duration=duration)
            chords = trim_grouped_chords_to_duration(chords, duration)
        except Exception as e:
            raise ValueError(f"[accordoai] Error during trimming/grouping chords: {str(e)}")

        try:
            # Step 2: Postprocess chords
            freq = get_chord_label_frequencies(chords)
            chords = handle_outlier_chords_by_label(df=chords, freq_counter=freq)
            chords = handle_misclassifications(df=chords)
            chords = normalize_chords(chords)
        except Exception as e:
            raise ValueError(f"[accordoai] Error during handling outlier chords: {str(e)}")

        return chords

    except Exception as e:
        raise ValueError(f'[accordoai] Error during postprocessing: {str(e)}')


def vector_to_chord(chordsdf):
    chords = get_chords()
    roots   = chords["roots"]
    basses  = chords["basses"]
    triads  = chords["triads"]
    fourths = chords["fourths"]
    
    chord_vocab_df = pd.read_csv(chord_vocab_file)
    
    # Apply smoothing to the predictions
    chordsdf = smooth_predictions(df=chordsdf)

    def process_row(row):
        try:
            # Extract indices
            root_idx = row['root']
            bass_idx = row['bass']
            triad_idx = row['triad']
            fourth_idx = row['fourth']
            
            # Validate and correct the vector
            vector = validate_and_correct_chord_vector([root_idx, bass_idx, triad_idx, fourth_idx])
            root_idx, bass_idx, triad_idx, fourth_idx = vector
            
            # Map indices to names
            root_val = roots[root_idx]
            bass_val = basses[bass_idx]
            triad_val = triads[triad_idx]
            fourth_val = fourths[fourth_idx]
            
            # Match with chord vocabulary
            chord_row = chord_vocab_df[
                (chord_vocab_df['root'] == root_val) &
                (chord_vocab_df['bass'] == bass_val) &
                (chord_vocab_df['triad'] == triad_val) &
                (chord_vocab_df['fourth'] == fourth_val)
            ]
            
            if chord_row.empty:
                return 'N'  # Return 'N' for unknown chords
            else:
                return chord_row['chord_name'].values[0]
        
        except Exception as e:
            raise ValueError(f"[accordoai] Error processing row: {str(e)}")

    try:
        # Apply the row-wise chord label generation
        chordsdf['chord_label'] = chordsdf.apply(process_row, axis=1)
        return chordsdf
    except Exception as e:
        raise ValueError(f"[accordoai] Error during chord label processing: {str(e)}")


def normalize_chords(df):
    def normalize_chord_label(label):
        
        if ':' in label:
            root, quality = label.split(':', 1)
        else:
            root, quality = label, ''  # no quality, raw note

        # Normalize enharmonic root
        root = ENHARMONIC_FLATS.get(root, root)

        return f"{root}:{quality}" if quality else root

    df['chord_label'] = df['chord_label'].apply(normalize_chord_label)
    return df




def group_chords(aligned_chords_df, beat_times, tempo, duration, pre_beat_percentage=0.4, post_beat_percentage=0):
    try:
        duration = time_str_to_seconds(duration)
        # Calculate the beat duration based on tempo (60 seconds / tempo in BPM)
        beat_duration = 60 / tempo

        grouped_chords = []

        for i in range(len(beat_times) - 1):
            # Calculate the current beat window
            start_time =float(beat_times[i])
            end_time = start_time + beat_duration
            
            # Calculate buffer time relative to the beat duration
            pre_beat_buffer_time = beat_duration * pre_beat_percentage
            post_beat_buffer_time = beat_duration * post_beat_percentage
            
            # Adjust the start time to be a bit before the beat starts (pre-buffer time)
            start_window = max(0, start_time - pre_beat_buffer_time)  # Prevent going below 0
            # Adjust the end window, considering the track duration for the last beat
            end_window = min(end_time + post_beat_buffer_time, duration) if i == len(beat_times) - 2 else end_time + post_beat_buffer_time
            
            # Find chords that fall within this expanded time window
            chords_in_window = aligned_chords_df[(aligned_chords_df['timestep'] >= float(start_window)) & 
                                                (aligned_chords_df['timestep'] < float(end_window))]
            
            # If there are chords in the window, take the most frequent one or average
            if len(chords_in_window) > 0:
                most_frequent_chord = chords_in_window['chord_label'].mode()[0]  # Taking the most frequent chord
                grouped_chords.append({
                    'chord_label': most_frequent_chord,
                    'timestep': start_time,
                })

        return pd.DataFrame(grouped_chords)
    except Exception as e:
        raise ValueError(f"[accordoai] Error during grouping chords: {str(e)}")



def trim_grouped_chords_to_duration(grouped_chords_df, duration):
    try:
        duration = time_str_to_seconds(duration)
        # Check which rows are within the song duration based on the beat start time
        mask = grouped_chords_df['timestep'] <= duration

        if not mask.all():
            # Find the first row where start_time exceeds duration
            first_exceed_idx = mask.idxmin()
            # Keep all rows before this point
            trimmed_df = grouped_chords_df.iloc[:first_exceed_idx].copy()
        else:
            trimmed_df = grouped_chords_df.copy()

        return trimmed_df
    except Exception as e:
        raise ValueError(f"[accordoai] Error during trimming grouped chords: {str(e)}")
    
    
    
def time_str_to_seconds(time_str):
    if isinstance(time_str, (int, float)):
        return float(time_str)
    if isinstance(time_str, str) and ':' in time_str:
        minutes, seconds = map(int, time_str.strip().split(':'))
        return minutes * 60 + seconds
    return float(time_str) 



def smooth_predictions(df, window_size=5, fourth_window_size=10):
    smoothed_chords = []
    try:
        for i in range(len(df)):
            smoothed_vector = []

            # --------------------------
            # Prepare main window
            try:
                start = max(0, i - window_size)
                end = min(len(df), i + window_size + 1)
                window = [
                    vec if isinstance(vec, (list, np.ndarray)) and len(vec) == 4 else [0, 0, 0, 0]
                    for vec in df['Chord_vector'].iloc[start:end].tolist()
                ]
                window_array = np.array(window)
            except Exception as e:
                raise ValueError(f"[accordoai][Window preparation error] at row {i}: {e}")
            
            # --------------------------
            # Smooth root (index 0)
            try:
                root_mode = stats.mode(window_array[:, 0], axis=None)[0]
                if isinstance(root_mode, np.ndarray):
                    root_mode = root_mode.item()  # Extract scalar if it's an array
                smoothed_vector.append(int(root_mode))
            except Exception as e:
                raise ValueError(f"[accordoai][Smoothing root error] at row {i}: {e}")
            
            # --------------------------
            # Smooth bass (index 1)
            try:
                bass_mode = stats.mode(window_array[:, 1], axis=None)[0]
                if isinstance(bass_mode, np.ndarray):
                    bass_mode = bass_mode.item()
                smoothed_vector.append(int(bass_mode))
            except Exception as e:
                raise ValueError(f"[accordoai][Smoothing bass error] at row {i}: {e}")

            # --------------------------
            # Smooth triad (index 2)
            try:
                triad_mode = stats.mode(window_array[:, 2], axis=None)[0]
                if isinstance(triad_mode, np.ndarray):
                    triad_mode = triad_mode.item()
                smoothed_vector.append(int(triad_mode))
            except Exception as e:
                raise ValueError(f"[accordoai][Smoothing triad error] at row {i}: {e}")

            # --------------------------
            # Smooth fourth (index 3) — using different window
            try:
                start_fourth = max(0, i - fourth_window_size)
                end_fourth = min(len(df), i + fourth_window_size + 1)
                window_fourth = [
                    vec if isinstance(vec, (list, np.ndarray)) and len(vec) == 4 else [0, 0, 0, 0]
                    for vec in df['Chord_vector'].iloc[start_fourth:end_fourth].tolist()
                ]
                window_fourth_array = np.array(window_fourth)
                fourth_mode = stats.mode(window_fourth_array[:, 3], axis=None)[0]
                if isinstance(fourth_mode, np.ndarray):
                    fourth_mode = fourth_mode.item()
                smoothed_vector.append(int(fourth_mode))
            except Exception as e:
                raise ValueError(f"[accordoai][Smoothing fourth error] at row {i}: {e}")
            
            # Append final smoothed vector
            smoothed_chords.append(smoothed_vector)

        df['Chord_vector'] = smoothed_chords
        # print(df)
        return df

    except Exception as e:
        raise ValueError(f"[accordoai][Smooth prediction failed]: {e}")



def is_valid_chord(root, bass, triad, fourth):
    # Handle cases where root, bass, or triad is "X" or "N"
    if (root == 'X' and triad == 'X') or (root == 'X' and triad == 'N') or (root == 'N' and triad == 'X') or (root == 'N' and triad == 'N'):
        return True  # Invalid chord if root or triad has invalid "X" or "N" combination
    
    if (root == 'X' and bass == 'X') or (root == 'N' and bass == 'X') or (root == 'X' and bass == 'N'):
        return True  # Invalid chord if root or bass has invalid "X" or "N" combination
    
    if(root == 'X' or triad == 'X' or root == 'N' or triad == 'N'):
        return True
    
    # Fourth can be "X" or "N" and doesn't affect validity
    # If we reach here, the chord might still be valid
    return False



def validate_and_correct_chord_vector(chord_vector):
    # print("DEBUG: chord_vector:", chord_vector)
    root_idx, bass_idx, triad_idx, fourth_idx = chord_vector
    
    chords = get_chords()
    roots   = chords["roots"]
    basses  = chords["basses"]
    triads  = chords["triads"]
    fourths = chords["fourths"]
    
    root = roots[root_idx]  # Convert root index to note
    bass = basses[bass_idx]  # Convert bass index to note
    triad = triads[triad_idx]  # Convert triad index to chord type
    fourth = fourths[fourth_idx]  # Convert fourth index to chord extension
    
    chord_vocab_df = pd.read_csv(chord_vocab_file)
    
    # Match with chord vocabulary
    chord_row = chord_vocab_df[
        (chord_vocab_df['root'] == root) &
        (chord_vocab_df['bass'] == bass) &
        (chord_vocab_df['triad'] == triad) &
        (chord_vocab_df['fourth'] == fourth)
    ]
    
    if chord_row.empty:
        label =  'N'  # Return 'N' for unknown chords
    else:
        label = chord_row['chord_name'].values[0]

    if label == 'N' or label == 'X':
        if not is_valid_chord(root, bass, triad, fourth):
            return correct_invalid_chord(root, bass, triad, fourth)
        else :
            return chord_vector
    else:
        return chord_vector
    
    
# Function to correct the entire chord vector
def correct_invalid_chord(root, bass, triad, fourth):
    # Correct invalid bass using the defined intervals
    bass = correct_invalid_bass(root, bass, triad)
    
    chords = get_chords()
    roots   = chords["roots"]
    basses  = chords["basses"]
    triads  = chords["triads"]
    fourths = chords["fourths"]
    
    r = roots.index(root)  # Get the index of the root
    b = basses.index(bass)  # Get the
    t = triads.index(triad)  # Get the index of the triad
    f = fourths.index(fourth)  # Get the index of the fourth
    
    vector = [r, b, t, f]  # Create the corrected chord vector]
    # print("DEBUG: Corrected chord vector:", vector)
    return vector


# Function to get the index of a note in the chromatic scale
def get_note_index(note):
    return chromatic_scale.index(note) if note in chromatic_scale else -1


# Function to get the note corresponding to a given index (semitone)
def get_note_from_index(index):
    return chromatic_scale[index % 12]  # Wrap around to 12 notes if necessary


# Function to correct invalid bass based on the triad and root
def correct_invalid_bass(root, bass, triad):
    # Define semitone intervals for each triad type
    intervals = {
        "Major": [0, 5, 7],    # Root, 4th (5 semitones), 5th (7 semitones)
        "Minor": [0, 3, 7],    # Root, 3rd (3 semitones), 5th (7 semitones)
        "Diminished": [0, 3, 7],  # Root, 3rd (3 semitones), 5th (7 semitones)
        "Augmented": [0, 5, 8],   # Root, 4th (5 semitones), 5th (8 semitones)
    }
    
    # Get the index of the root
    root_index = get_note_index(root)
    
    # If the bass is invalid ("X" or "N"), we need to choose a valid bass
    if bass == "X" or bass == "N":
        # Get the valid intervals based on the triad type
        allowed_intervals = intervals.get(triad, [0])  # Default to root if triad is not found
        
        # Try to find a valid bass by applying each interval
        for interval in allowed_intervals:
            bass_index = root_index + interval
            bass = get_note_from_index(bass_index)  # Get the note from the calculated index
            
    return bass



def get_chord_label_frequencies(df):
    chord_labels = df['chord_label'].tolist()
    freq_counter = Counter(chord_labels)
    return freq_counter



def handle_outlier_chords_by_label(df, freq_counter, threshold=20, min_consecutive=2, w=5):
    chord_list = df['chord_label'].tolist()
    length = len(chord_list)
    print(freq_counter)

    def is_rare(label):
        return freq_counter.get(label, 0) <= threshold

    def get_neighbor_mode(i):
        window = chord_list[max(0, i-w):min(length, i+1+w)]
        filtered = [label for label in window if not is_rare(label)]
        if not filtered:
            return chord_list[i - 1] if i > 0 else 'N'
        return Counter(filtered).most_common(1)[0][0]

    i = 0
    while i < length:
        current = chord_list[i]
        if is_rare(current):
            j = i
            while j < length and chord_list[j] == current:
                j += 1
            repeat_count = j - i
            if repeat_count >= min_consecutive:
                replacement = get_neighbor_mode(i)
                for k in range(i, j):
                    chord_list[k] = replacement
            i = j
        else:
            i += 1

    df['chord_label'] = chord_list
    return df




def chromatic_distance(chord1, chord2):
    pc1 = set(p.midi % 12 for p in chord1.pitches)
    pc2 = set(p.midi % 12 for p in chord2.pitches)
    return len(pc1.symmetric_difference(pc2))



def handle_misclassifications(df, window_size=3, freq_threshold=10, chromatic_threshold=2):

    chord_list = df['chord_label'].tolist()
    corrected_chords = []

    # Step 1: Calculate frequency of each chord in the song
    chord_freq = Counter(chord_list)
    total_chords = len(chord_list)
    chord_freq_percent = {k: v / total_chords for k, v in chord_freq.items()}

    for i in range(len(chord_list)):
        current_label = chord_list[i]

        try:
            current_chord = chord.Chord(current_label)
        except:
            # If parsing fails, keep as is
            corrected_chords.append(current_label)
            continue

        # Create window of neighbors
        start = max(0, i - window_size)
        end = min(len(chord_list), i + window_size + 1)
        window_labels = chord_list[start:i] + chord_list[i+1:end]  # exclude current

        close_neighbors = []
        for neighbor_label in window_labels:
            try:
                neighbor_chord = chord.Chord(neighbor_label)
                dist = chromatic_distance(current_chord, neighbor_chord)
                if dist <= chromatic_threshold:
                    close_neighbors.append(neighbor_label)
            except:
                continue

        # If there are no close neighbors OR the chord is rare → correct it
        freq = chord_freq_percent.get(current_label, 0)

        if not close_neighbors or freq < freq_threshold:
            # Choose the most common chord in the window
            candidate_chords = window_labels + close_neighbors
            if candidate_chords:
                new_chord = Counter(candidate_chords).most_common(1)[0][0]
                corrected_chords.append(new_chord)
            else:
                corrected_chords.append(current_label)
        else:
            corrected_chords.append(current_label)

    df['chord_label'] = corrected_chords
    return df
