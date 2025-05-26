import tensorflow as tf


def load():
    # Load the trained model
    model_path = 'models/'
    model = tf.keras.models.load_model(model_path, compile=False)
    return model
