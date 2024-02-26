import tensorflow as tf
from tensorflow.keras.models import load_model

# Path to your '.h5' model
model = load_model('D:/1 PraTA/Code/CNN/model.h5')
# Path where the TensorFlow Lite model will be saved
tflite_model_path = 'D:/1 PraTA/Code/CNN/model.tflite'
# Path for the quantized model, if you choose to use quantization
tflite_quant_model_path = 'model_quant.tflite'

# Load the Keras model
# model = load_model(h5_model_path)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TensorFlow Lite model saved at: {tflite_model_path}')


# Uncomment below code if you want to apply float16 quantization

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_quant_model = converter.convert()

# Save the quantized TensorFlow Lite model
# with open(tflite_quant_model_path, 'wb') as f:
#     f.write(tflite_quant_model)

# print(f'Quantized TensorFlow Lite model saved at: {tflite_quant_model_path}')
