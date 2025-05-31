import tensorflow as tf
import numpy as np
import pandas as pd
from .config import get_config

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
    print("[accordoai] Starting prediction...")
    try:
        if features is None or len(features) == 0:
            raise ValueError("[accordoai] No features provided for prediction.")
        
        print(f"[accordoai] Features shape: {features.shape}, Sequence length: {seq_length}, Hops: {hops}, Sample Rate: {sr}")

        for chunk_idx in range(features.shape[0]):
            chunk = features[chunk_idx]                    
            chunk = np.expand_dims(chunk, axis=0)             
            
            # if chunk_idx == 0:
            #     print(model.summary())

            # Predict
            predictions = model.predict(chunk, verbose=0)

            # Get predicted classes
            root_preds   = np.argmax(predictions[0], axis=-1).flatten()
            bass_preds   = np.argmax(predictions[1], axis=-1).flatten()
            triad_preds  = np.argmax(predictions[2], axis=-1).flatten()
            fourth_preds = np.argmax(predictions[3], axis=-1).flatten()

            for j in range(300):
                timestep = chunk_idx * 300 + j
                time_in_seconds = timestep * time_per_timestep

                root_pred   = int(root_preds[j])
                bass_pred   = int(bass_preds[j])
                triad_pred  = int(triad_preds[j])
                fourth_pred = int(fourth_preds[j])

                # print(f"Timestep {timestep} ({time_in_seconds:.2f} s): "
                #     f"[Root: {root_pred}, Bass: {bass_pred}, Triad: {triad_pred}, Fourth: {fourth_pred}]")

                predicted_chord_vector.append([
                    round(time_in_seconds, 4),
                    [root_pred, bass_pred, triad_pred, fourth_pred]
                ])

        # Build final DataFrame
        chordsdf = pd.DataFrame(predicted_chord_vector, columns=['timestep', 'Chord_vector'])
        return chordsdf
    except Exception as e:
        print(f"[accordoai] Prediction error: {e}")
        return None