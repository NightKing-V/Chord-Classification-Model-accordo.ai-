from importlib.resources import files
from keras.models import load_model



def load():
    model_path = files("accordoai.models").joinpath("model.keras")
    print(f"[accordoai] Loading model...")
    model = load_model(str(model_path))
    return model
