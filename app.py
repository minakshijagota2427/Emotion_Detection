# Emotion Detection AI App
from cProfile import label
from transformers import pipeline
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
    color: white;
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
/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;   /* dark navy */
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* RADIO BUTTONS */
div[role="radiogroup"] label {
    color: #cbd5e1 !important;
    font-weight: 500;
}

/* SELECTED RADIO */
div[role="radiogroup"] label[data-checked="true"] {
    color: #22c55e !important;   /* green highlight */
}

/* SIDEBAR TITLE */
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    padding: 8px 15px;
    background-color: RED;
    color: black;
    border: none;
}
/* 🔥 SEPARATOR STYLE (YAHAN ADD KARNA HAI) */
section[data-testid="stSidebar"] hr {
    border-color: #374151;
}
/* 🔥 BLUR OVERLAY */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;

    backdrop-filter: blur(1px);   /*  blur level */
    background: rgba(0, 0, 0, 0.7); /*  dark overlay */

    z-index: 0;
}

/* 🔥 KEEP CONTENT ABOVE */
.stApp > * {
    position: relative;
    z-index: 1;
}
/* TEXT INPUT LABEL FIX */
label[data-testid="stWidgetLabel"] {
    color: white !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
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
import re
from collections import deque
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

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text
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
# MODEL_PATH = "model/emotion_model.keras" 
# FILE_ID = "1gmQcc6tkLwhi3rsiIyqPiBItcKT61jZU"   #  actual ID
# DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
# if not os.path.exists(MODEL_PATH):
#     with st.spinner("Downloading model... ⏳"):
#         gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
# # 🔥 STEP 5: LOAD MODEL
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)

# face_model = load_model()
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
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-go-emotion"
    )

emotion_classifier = load_emotion_model()
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
class EmotionProcessor(VideoProcessorBase):

    def __init__(self):
        self.emotion_buffer = deque(maxlen=10)   # last 10 frames
        self.latest_emotion = None
        self.frame_count = 0
        self.latest_confidence = 0
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
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB) 
            face = face / 255.0
            face = np.reshape(face, (1,48,48,3))
            pred = face_model(face, training=False).numpy()

            label = emotion_labels[np.argmax(pred)]
            confidence = float(np.max(pred))

            # most frequent emotion (smooth output)
            self.latest_emotion = max(set(self.emotion_buffer), key=self.emotion_buffer.count)
            self.latest_confidence = confidence
            if confidence > 0.6:
                self.emotion_buffer.append(label)

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
        <h3> Total Predictions</h3>
        <h2 style="color:#00ffcc; font-size:28px; margin-top:0.2px; ">{total}</h2>
        </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown(f"""
        <div class="card">
        <h3> Last Emotion</h3>
        <h2 style="color:#00ffcc; font-size:28px; margin-top:30px;">{last}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card">
        <h3> Status</h3>
        <h2 style="color:#00ffcc; font-size:28px; margin-top:30px;">Active</h2>
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

    #  convert to RGB
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
        result = emotion_classifier(text)

        pred = result[0]['label']
        confidence = result[0]['score']

        color_map = {
                    "Happy": "#22c55e",
                    "Sad": "#3b82f6",
                    "Angry": "#ef4444",
                    "Fear": "#f59e0b",
                    "Surprise": "#a855f7",
                    "Neutral": "#9ca3af"
                }

        color = color_map.get(pred, "#22c55e")

        st.markdown(f"""
                <div style="
                    background: rgba(0,0,0,0.75);
                    padding: 18px;
                    border-radius: 12px;
                    text-align: center;
                    font-size: 22px;
                    font-weight: bold;
                    color: {color};
                    margin-top: 15px;
                ">
                    Emotion: {pred}
                </div>
                """, unsafe_allow_html=True)
        save_prediction("Text", pred)
    st.markdown("---")
    
    # 🔹 Image upload
    uploaded_img = st.file_uploader("Upload Text Image", type=["png","jpg","jpeg"])

    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image)

        extracted_text = pytesseract.image_to_string(image)

        st.subheader("Extracted Text:")
        st.write(extracted_text)

        if extracted_text.strip():
            cleaned = clean_text(extracted_text)

            X = vectorizer.transform([cleaned])
            pred = text_model.predict(X)[0]
       
            st.markdown(f"""
                <div style="
                    background: rgba(0, 0, 0, 0.6);
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 20px;
                    font-weight: 600;
                    color: #22c55e;
                ">
                    Emotion: {pred}
                </div>
                """, unsafe_allow_html=True)
            save_prediction("Image Text", pred)
# -------- LIVE CAMERA (NO WEBRTC) --------
elif page == "Live Detection":

    st.header("🎥 Live Emotion Detection ")

    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Camera"):
            st.session_state.camera_on = True

    with col2:
        if st.button("⛔ Stop Camera"):
            st.session_state.camera_on = False

    FRAME_WINDOW = st.image([])

    if st.session_state.camera_on:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Camera not accessible")
        else:
            st.success("✅ Camera running")

        from collections import deque
        emotion_buffer = deque(maxlen=10)

        while st.session_state.camera_on:

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 3)

            for (x, y, w, h) in faces[:1]:

                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48,48))
                face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

                face = face / 255.0
                face = np.reshape(face, (1,48,48,3))

                pred = face_model(face, training=False).numpy()

                label = emotion_labels[np.argmax(pred)]
                confidence = float(np.max(pred))

                if confidence > 0.6:
                    emotion_buffer.append(label)

                if len(emotion_buffer) > 0:
                    smooth_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
                else:
                    smooth_emotion = label
                confidence_percent = confidence * 100
                # DRAW
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{smooth_emotion} ({confidence_percent:.1f}%)",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,0), 2)

                # SAVE (no spam)
                if smooth_emotion != st.session_state.last_saved_emotion:
                    save_prediction("Live", smooth_emotion)
                    st.session_state.last_saved_emotion = smooth_emotion

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

