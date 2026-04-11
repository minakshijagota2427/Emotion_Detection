# Emotion Detection AI App
import streamlit as st

# 🔥 FIRST command
st.set_page_config(page_title="Emotion AI", layout="centered")

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import os
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import os
import tensorflow as tf
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ---------------- PATH FIX (IMPORTANT) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

face_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model/emotion_model.keras")
)
text_model = pickle.load(open(os.path.join(BASE_DIR, "model/text_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "model/vectorizer.pkl"), "rb"))

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))

            # 🔥 RGB fix (VERY IMPORTANT)
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

            face = face / 255.0
            face = np.reshape(face, (1,48,48,3))

            pred = face_model.predict(face, verbose=0)
            label = emotion_labels[np.argmax(pred)]

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, label,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- LIVE CAMERA CLASS ----------------
st.header("🎥 Capture Emotion from Camera")

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if st.button("Start Camera"):
    st.session_state.camera_on = True
if st.button("Stop Camera"):
    st.session_state.camera_on = False

if st.session_state.camera_on:
    camera_image = st.camera_input("Take a photo")

    if camera_image:
        image = Image.open(camera_image)
        img = np.array(image)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

            face = face / 255.0
            face = np.reshape(face, (1,48,48,3))

            pred = face_model.predict(face, verbose=0)
            label = emotion_labels[np.argmax(pred)]
            confidence = np.max(pred)

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,f"{label} ({confidence:.2f})",
                        (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(255,0,0),2)

        st.image(img, caption="Detected Emotion")

# if camera_image:
#     image = Image.open(camera_image)
#     img = np.array(image)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x,y,w,h) in faces:
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (48,48))

# 🔥 IMPORTANT FIX
    #     face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

    #     face = face / 255.0
    #     face = np.reshape(face, (1,48,48,3))

    #     pred = face_model.predict(face, verbose=0)
    #     label = emotion_labels[np.argmax(pred)]
    #     confidence = np.max(pred)

    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #     cv2.putText(img,f"{label} ({confidence:.2f})",
    #                 (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.9,(255,0,0),2)

    # st.image(img, caption="Detected Emotion")
# ---------------- UI ----------------

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

# 👉 convert to RGB
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

        face = face / 255.0
        face = np.reshape(face, (1,48,48,3))

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
# session state
if "live_on" not in st.session_state:
    st.session_state.live_on = False

# buttons
if st.button("Start Live Detection"):
    st.session_state.live_on = True

if st.button("Stop Live Detection"):
    st.session_state.live_on = False

# live camera
if st.session_state.live_on:
    webrtc_streamer(
        key="live-emotion",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )