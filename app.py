# Emotion Detection AI App
import streamlit as st
import base64
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)
# 🔥 FIRST command
st.set_page_config(page_title="Emotion AI", layout="centered")
set_bg("bg.png")
# Sidebar
st.sidebar.markdown("## ⚡ Emotion AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "Home",
    "Face Detection",
    "Camera Detection",
    "Text Detection",
    "Live Detection",
    "Dashboard"
])
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: black;
}

/* File uploader fix */
[data-testid="stFileUploader"] {
    color: white !important;
}

[data-testid="stFileUploader"] label {
    color: white !important;
    font-weight: 500;
}

[data-testid="stFileUploader"] div {
    color: white !important;
}
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}
.card h3 {
    color: #ffffff !important;
    font-weight: 600;
}


/* Cards */
.card {
    background-color: #1c1f26;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    text-align: center;
}
.card h3 {
    color: #ffffff !important;
    font-weight: 600;
}

/*  CARD VALUE (NUMBER / TEXT) */
.card h1 {
    color: #ffffff !important;
}

/* Titles */
h1, h2, h3 {
    color: white;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    padding: 8px 15px;
    background-color: RED;
    color: black;
    border: none;
}

</style>
""", unsafe_allow_html=True)
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
import pandas as pd
from datetime import datetime
import time
# ---------------- SESSION INIT ----------------
from collections import deque

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Type", "Emotion"])

if "live_buffer" not in st.session_state:
    st.session_state.live_buffer = deque(maxlen=20)

if "last_saved_emotion" not in st.session_state:
    st.session_state.last_saved_emotion = None

if "last_save_time" not in st.session_state:
    st.session_state.last_save_time = 0
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Type", "Emotion"])
def save_prediction(pred_type, emotion):
    new = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Type": pred_type,
        "Emotion": emotion
    }
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([new])],
        ignore_index=True
    )
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ---------------- PATH FIX (IMPORTANT) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
})

face_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model/emotion_model.keras")
)
text_model = pickle.load(open(os.path.join(BASE_DIR, "model/text_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "model/vectorizer.pkl"), "rb"))

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
class EmotionProcessor(VideoProcessorBase):

    def __init__(self):
        self.latest_emotion = None
        self.latest_confidence = None
        self.frame_count = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # resize for speed
        img = cv2.resize(img, (640, 480))

        self.frame_count += 1

        # skip frames
        if self.frame_count % 5 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces[:1]:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))

            face = face / 255.0
            face = np.reshape(face, (1,48,48,1))

            pred = face_model(face, training=False).numpy()

            label = emotion_labels[np.argmax(pred)]
            confidence = float(np.max(pred))

            self.latest_emotion = label
            self.latest_confidence = confidence

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f"{label}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
# ---------------- UI ----------------

if page == "Home":
    st.title(" Emotion AI Dashboard")

    col1, col2, col3 = st.columns(3)

    total = len(st.session_state.history)
    last = st.session_state.history["Emotion"].iloc[-1] if total > 0 else "None"

    with col1:
        st.markdown(f"""
        <div class="card">
        <h4> Total Predictions</h4>
        <h2 style="color:#00ffcc; font-size:28px; margin-top:1px;">{total}</h2>
        </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown(f"""
        <div class="card">
        <h3> Last Emotion</h3>
        <h2 style="color:#00ffcc; font-size:28px; margin-top:14px;">{last}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card">
        <h3> Status</h3>
        <h2 style="color:#00ffcc; font-size:28px; margin-top:14px;">Active</h2>
        </div>
        """, unsafe_allow_html=True)

# ---------------- LIVE CAMERA CLASS ----------------
elif page == "Camera Detection":
    st.title("Camera Emotion Detection")

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
            label = None
            confidence = None
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
            if label:
                save_prediction("Camera", label)
            else:
                st.warning("No face detected")

# -------- FACE --------
elif page == "Face Detection":
    st.title("Face Emotion Detection")

    uploaded_file = st.file_uploader("Upload Image")

    if uploaded_file:
        image = Image.open(uploaded_file)
        img = np.array(image)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        label = None
        confidence = None
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
                        0.6,(255,0,0),2)

        st.image(img, caption="Detected Emotion")
        save_prediction("Face", label)


# -------- TEXT --------
elif page == "Text Detection":
    st.title("Text Emotion Detection")

    text = st.text_input("Enter text")

    if text:
        X = vectorizer.transform([text])
        pred = text_model.predict(X)[0]

        st.success(f"Emotion: {pred}")
        save_prediction("Text", pred)
# -------- LIVE CAMERA (NO WEBRTC) --------
elif page == "Live Detection":

    st.header("🎥 Live Emotion Detection (Local Camera)")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not accessible")
        else:
            st.success("Camera started")

        while run:
            ret, frame = cap.read()

            if not ret:
                st.error("Failed to capture frame")
                break

            # -------- PROCESS FRAME --------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces[:1]:

                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48,48))

                face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # ✅ convert to 3 channel
                face = face / 255.0
                face = np.reshape(face, (1,48,48,3))          # ✅ correct shape

                pred = face_model(face, training=False).numpy()

                label = emotion_labels[np.argmax(pred)]
                confidence = float(np.max(pred))

                # DRAW
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,0), 2)

                # SAVE
                save_prediction("Live", label)

            # SHOW FRAME
            FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()


# dashboard
elif page == "Dashboard":
    st.title("📊 Analytics Dashboard")

    df = st.session_state.history
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

    st.subheader("📈 Emotion Timeline")

    timeline = df.groupby("Time").count()["Emotion"]

    st.line_chart(timeline)

    if len(df) == 0:
        st.warning("No data yet")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Emotion Distribution")
            if len(df) > 0:
                st.bar_chart(df["Emotion"].value_counts())
            st.markdown('</div>', unsafe_allow_html=True)


        with col2:
            st.subheader("Emotion Pie Chart")
            st.write(
                df["Emotion"].value_counts().plot.pie(autopct="%1.1f%%").figure
            )
        st.markdown("### 📜 History")
        st.dataframe(df, use_container_width=True)

        most = df["Emotion"].value_counts().idxmax()

        st.markdown(f"""
        <div class="card">
        <h3> Most Frequent Emotion</h3>
        <h1 style="color:#00ffcc; font-size:28px; margin-top:10px;">{most}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("History")
        st.dataframe(df)

        most = df["Emotion"].value_counts().idxmax()
        st.success(f"Most Frequent Emotion: {most}")

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "data.csv")

