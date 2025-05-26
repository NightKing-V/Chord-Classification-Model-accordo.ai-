import tensorflow as tf
import numpy as np
import pandas as pd
from config import get_config

# Process input in overlapping chunks of 300 time steps
predictions_root = []
predictions_bass = []
predictions_triad = []
predictions_fourth = []

predicted_chord_vector = []

config = get_config()
seq_length = config['SLICE_SIZE']
hops = config['HOP_LENGTH']
sr = config['SAMPLE_RATE']

time_per_timestep = hops / sr

def predict(model, features):

    for i in range(0, features.shape[0] - seq_length + 1, seq_length):
        chunk = features[i : i + seq_length]  # Take a 300-time-step chunk
        chunk = np.expand_dims(chunk, axis=0)  # Add batch dimension (1, 300, 12)
        
        # Predict using the model
        predictions = model.predict(chunk, verbose=0)
        
        # Store the predicted outputs
        predictions_root.append(np.argmax(predictions[0], axis=-1))
        predictions_bass.append(np.argmax(predictions[1], axis=-1))
        predictions_triad.append(np.argmax(predictions[2], axis=-1))
        predictions_fourth.append(np.argmax(predictions[3], axis=-1))

        # Convert predictions into a single array
        predicted_root = np.concatenate(predictions_root)
        predicted_bass = np.concatenate(predictions_bass)
        predicted_triad = np.concatenate(predictions_triad)
        predicted_fourth = np.concatenate(predictions_fourth)

        # print(i)
        
        for j in range(seq_length):
            timestep = i + j
            time_in_seconds = timestep * time_per_timestep  # Convert timestep index to seconds
            root_pred = int(predicted_root[0][j])
            bass_pred = int(predicted_bass[0][j])
            triad_pred = int(predicted_triad[0][j])
            fourth_pred = int(predicted_fourth[0][j])
        
            # Print the predictions for each timestep, along with the time in seconds
            print(f"Timestep {timestep} ({time_in_seconds:.2f} seconds): [Root: {root_pred}, Bass: {bass_pred}, Triad: {triad_pred}, Fourth: {fourth_pred}]")
            predicted_chord_vector.append([round(time_in_seconds, 4), [root_pred, bass_pred, triad_pred, fourth_pred]])
    
    chordsdf = pd.DataFrame(predicted_chord_vector, columns=['timestep', 'Chord_vector'])
    return chordsdf