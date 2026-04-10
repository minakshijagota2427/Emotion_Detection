import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from streamlit_webrtc import RTCConfiguration
# ---------------- PATH FIX (IMPORTANT) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

face_model = load_model(os.path.join(BASE_DIR, "model/emotion_model.h5"))
text_model = pickle.load(open(os.path.join(BASE_DIR, "model/text_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "model/vectorizer.pkl"), "rb"))

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------------- LIVE CAMERA CLASS ----------------
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))
            face = face / 255.0
            face = np.reshape(face, (1,48,48,1))

            pred = face_model.predict(face, verbose=0)
            label = emotion_labels[np.argmax(pred)]
            confidence = np.max(pred)

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f"{label} ({confidence:.2f})",
                        (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- UI ----------------
st.set_page_config(page_title="Emotion AI", layout="centered")

st.title("😃 Emotion Detection AI App")

# -------- FACE --------
st.header("📸 Face Emotion Detection")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        pred = face_model.predict(face, verbose=0)
        label = emotion_labels[np.argmax(pred)]
        confidence = np.max(pred)

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,f"{label} ({confidence:.2f})",
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(255,0,0),2)

    st.image(img, caption="Detected Emotion")


# -------- TEXT --------
st.header("🧠 Text Emotion Detection")

text = st.text_input("Enter text")

if text:
    X = vectorizer.transform([text])
    pred = text_model.predict(X)[0]

    st.success(f"Emotion: {pred}")


# -------- LIVE CAMERA --------
st.header("🎥 Live Emotion Detection (Camera)")

try:
    webrtc_streamer(
        key="emotion",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
except Exception as e:
    st.error("Camera not supported in this environment. Try running locally.")