import tensorflow as tf

def convert_to_tflite(keras_model_path, tflite_model_path):
    # Load the Keras model
    model = tf.keras.models.load_model(keras_model_path)

    # Convert the model to a TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to disk
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    keras_model_path = "Models/mask_detection_192.keras"  # Path to the Keras model
    tflite_model_path = "Models/mask_detection_192.tflite"  # Path to save the TFLite model

    convert_to_tflite(keras_model_path, tflite_model_path)
