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

            features, duration = preprocess(path=file_path)
            vectors = predict(model=self.model, features=features)
            chords = postprocess(df=vectors, duration=duration)

            return chords

        except Exception as e:
            print(f"[accordoai] Error: {e}")
            return None

        finally:
            print("[accordoai] Prediction completed.")
