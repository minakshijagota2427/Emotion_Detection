import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("emotion_model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load image
img = cv2.imread("test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48,48))
    # convert grayscale → RGB
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    face = face / 255.0
    face = np.reshape(face, (1,48,48,3))

    prediction = model.predict(face, verbose=0)
    emotion = emotion_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    cv2.rectangle(img, (x, y-30), (x+150, y), (0,0,0), -1)

    cv2.putText(img, f"{emotion} ({confidence:.2f})",
            (x+5,y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2)

# Save result
cv2.imwrite("output.jpg", img)

print("✅ Done! Check output.jpg")