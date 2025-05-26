import os
from .preprocessor import preprocess
from .model_loader import load
from .predictor import predict
from .postprocessor import postprocess

class ChordPredictor:
    def __init__(self):
        self.model = load()  # Load model once at init

    def predict_chords(self, file_path: str):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"[accordoai] Incorrect file path: {file_path}")

            features, duration = preprocess(file_path)
            vectors = predict(self.model, features)
            chords = postprocess(vectors, duration)

            return chords

        except Exception as e:
            print(f"[accordoai] Error: {e}")
            return None

        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[accordoai] Deleted processed file: {file_path}")
            except Exception as e:
                print(f"[accordoai] Failed to delete file: {e}")
