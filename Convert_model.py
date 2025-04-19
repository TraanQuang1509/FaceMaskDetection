import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model("best_model.keras")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: giảm size và tăng tốc
tflite_model = converter.convert()

# Save
with open("best_model_lite.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Đã chuyển sang TFLite thành công!")
