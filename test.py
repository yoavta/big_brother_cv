# import tensorflow as tf
#
# saved_model_dir = 'Resources/ssd_mobilenet_v2_2/'
# tflite_model_path = 'Resources/ssd_mobilenet_v2_2/ssd_mobilenet_v2.tflite'
#
# # Load the model
# model = tf.saved_model.load(saved_model_dir)
#
# # Convert the model to TFLite format
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()
#
# # Save the TFLite model
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_model)
#
# print(f"Model converted to TFLite format and saved as {tflite_model_path}")
