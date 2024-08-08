from tensorflow.keras.models import load_model

def load_trained_model(model_path):
    """ Load the trained Keras model from the specified path. """
    return load_model(model_path)

if __name__ == "__main__":
    model = load_trained_model('model/dqn_trading_model.h5')
    print("Model loaded successfully.")
