import tensorflow as tf

model = tf.keras.models.load_model(
    "model/emotion_model.h5",
    compile=False
)

model.save("model/emotion_model.keras")

print("Model converted successfully ✅")
