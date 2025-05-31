import os
import librosa
import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from .config import get_config
import filetype
import os
# from pathlib import Path
# import shutil

# Function to preprocess audio file and extract features
def preprocess(path):
    print(f"[accordoai] Starting preprocessing...")
    try:
        config = get_config()
        sr = config['SAMPLE_RATE']
        bins = config['BINS_PER_OCT']
        slice_size = config['SLICE_SIZE']
        mono = config['MONO']
        hops = config['HOP_LENGTH']

        saved_file, duration = convet_to_wav(path)
        
        cqt, time = feature_extraction(saved_file, sr, bins, mono, hops, slice_size)
        # beats, tempo = await detect_beats(saved_file)
        
        print(f"[accordoai] Preprocessed CQT shape: {cqt.shape}, Time shape: {time.shape}")

        # return cqt, beats, tempo
        return cqt, duration
    except Exception as e:
        print(f"[accordoai] An error occurred during preprocessing: {e}")
        return None, None, None
    finally:
        # Clean up the processed file if needed
        try:
            if os.path.exists(saved_file):
                os.remove(saved_file)
                print(f"[accordoai] Deleted processed file: {saved_file}")
        except Exception as e:
            print(f"[accordoai] Failed to delete processed file: {e}")


# Function to convert audio file to WAV format
def convet_to_wav(song_path):
    try:
        directory_path = os.path.dirname(song_path)
        file = os.path.basename(song_path) # Assuming the first file is the one to convert
        filename = os.path.splitext(file)[0]
        if is_valid_audio_file(file):
            new_song = os.path.join(directory_path, f"{filename}_conv.wav")
            audio = AudioFileClip(song_path)
            duration = audio.duration
            audio.write_audiofile(new_song, logger=None)
            print(f"[accordoai] Converted {song_path} to {new_song}")
            
            return new_song, duration
        else:
            raise Exception("[accordoai] Unsupported file format")
    except Exception as e:
        print(f"[accordoai] An error occurred: {e}")


# Function to extract features from audio file
def feature_extraction(file_path, sample_rate, bins, mono, hops, slice):
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, mono=mono)
        cqt = librosa.cqt(y, sr=sr, hop_length=2048, bins_per_octave=bins, n_bins=(bins*8))
        cqt = np.abs(cqt)

        octave_weights = np.array([1, 1.8, 1.8, 1.8, 2, 2.5, 3, 4])
        weights = np.ones(cqt.shape)

        for t in range(cqt.shape[1]):
            top_4_indices = np.argsort(cqt[:, t])[-4:]
            for idx in top_4_indices:
                octave = idx // 24
                weights[idx, t] = octave_weights[octave]

        cqt = cqt * weights
        timestamps = np.arange(cqt.T.shape[0]) * hops / sample_rate
        timestamps = np.round(timestamps, 4)

        cqt_array = cqt.T
        reshaped_array = restructure(cqt_array, slice, bins)
            
        print(f"[accordoai] Extracted features from {file_path}")
        return reshaped_array, timestamps
    except Exception as e:
        print(f"[accordoai] An error occurred during feature extraction: {e}")
        return None, None


# Function to restructure the extracted features
def restructure(data_array, seq_length, bins):
    try:
        num_rows = data_array.shape[0]
        num_columns = bins * 8
        remainder = num_rows % seq_length

        if remainder > 0:
            padding_needed = seq_length - remainder
            pad_array = np.zeros((padding_needed, num_columns))
            data_array = np.vstack([data_array, pad_array])

        # # Normalize the columns (if applicable)
        # data_array[:, :num_columns] = librosa.util.normalize(data_array[:, :num_columns], axis=0)

        reshaped_array = data_array.reshape(-1, seq_length, num_columns)

        return reshaped_array
    except Exception as e:
        print(f"[accordoai] An error occurred during restructuring: {e}")
        return None



def get_file_extension(file_bytes):
    kind = filetype.guess(file_bytes)

    if not kind:
        print("[accordoai] Could not detect file type.")
        return None

    mime_type = kind.mime
    print(f"[accordoai] Detected MIME Type: {mime_type}")

    # Check if MIME is in the allowed list before assigning extension
    allowed_mime_extensions = {
        'audio/mpeg': '.mp3',
        'audio/wav': '.wav',
        'audio/x-wav': '.wav',
        'audio/flac': '.flac',
        'audio/x-flac': '.flac',
        'audio/aac': '.aac',
        'audio/x-aac': '.aac',
        'audio/ogg': '.ogg',
        'audio/x-ogg': '.ogg',
        'audio/mp4': '.m4a',
        'audio/x-m4a': '.m4a',
        'audio/m4a': '.m4a',
        'audio/webm': '.webm',
        'audio/x-ms-wma': '.wma'
    }

    extension = allowed_mime_extensions.get(mime_type)

    if extension:
        print(f"[accordoai] Detected Extension: {extension}")
        return extension
    else:
        print("[accordoai]Unsupported MIME type")
        return None


# Function to detect valid audio file extensions
def is_valid_audio_file(file):
    valid_extensions = ('.mp3', '.flac', '.wav', '.ogg', '.aac', '.wma', '.m4a', '.webm')
    return file.endswith(valid_extensions)
    
    
# Function to detect beats in an audio file
def detect_beats(audio_path):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path)

        # Detect the beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, tightness=90, trim=True)

        # Convert frame indices to time (in seconds)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        return beat_times, float(tempo[0])
    except Exception as e:
        print(f"[accordoai] An error occurred during beat detection: {e}")
        return None, None
    
    

