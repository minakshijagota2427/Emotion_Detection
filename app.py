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
st.set_page_config(page_title="Emotion AI",page_icon="🧠", layout="wide")
set_bg("bg.png")
# Sidebar
st.sidebar.markdown(
    """
    <div class='title'>⚡ Emotion AI</div>
    <div class='sub'>Smart Emotion Recognition</div>
    <hr>
    """,
    unsafe_allow_html=True
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

menu = [
    ("🏠 Home", "Home"),
    ("😊 Face Detection", "Face Detection"),
    ("📷 Camera Detection", "Camera Detection"),
    ("💬 Text Detection", "Text Detection"),
    ("🎥 Live Detection", "Live Detection"),
    ("📊 Dashboard", "Dashboard"),
]

for btn_text, page_name in menu:
    active = st.session_state.page == page_name

    label = f"🟢 {btn_text}" if active else btn_text

    if st.sidebar.button(
        label,
        use_container_width=True,
        key=page_name
    ):
        st.session_state.page = page_name
page = st.session_state.page
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
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;

    backdrop-filter: blur(25px);          /* increased from 10px → 25px */
    -webkit-backdrop-filter: blur(25px);

    background: rgba(0, 0, 0, 0.65);      /* increased from 0.4 → 0.65 */

    z-index: -1;
    pointer-events: none;
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
    background: linear-gradient(180deg,#071326,#0b1d39,#10294d);
    background-color: #111827 !important;   /* dark navy */
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: white !important;
}
.title{
    font-size:28px;
    font-weight:700;
    color:white;
    margin-bottom:8px;
}

.sub{
    color:#b7c5dd;
    margin-bottom:25px;
}

div.stButton > button{
    width:100%;
    text-align:left;
    padding:14px 18px;
    border-radius:14px;
    border:none;
    margin-bottom:10px;
    background: rgba(255,255,255,0.07);
    color:white;
    font-size:17px;
    font-weight:500;
    transition:0.3s;
}
.glass-box{
    background: rgba(0,0,0,0.45);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 30px;
    margin-top: 20px;
    box-shadow: 0 8px 35px rgba(0,0,0,.35);
    border: 1px solid rgba(255,255,255,.12);
}

/* ── NEW: full-width info card ── */
.info-card {
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px 28px;
    margin-top: 16px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.10);
    color: white;
}

.info-card h3 {
    color: black !important;
    font-weight: 600;
    margin-bottom: 6px;
}

.info-card p, .info-card li {
    color: #d1d5db !important;
    font-size: 15px;
}

/* RADIO BUTTONS */

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
div.stButton > button{
    width:100%;
    text-align:left;
    padding:14px 18px;
    border-radius:14px;
    border:none;
    margin-bottom:10px;
    background: black;
    color:white;
    font-size:17px;
    font-weight:500;
    transition:.3s;
}

div.stButton > button:hover{
    background: linear-gradient(90deg,#ff4b4b,#ff6b6b);
    transform: translateX(5px);
    box-shadow:0 0 20px rgba(255,75,75,.35);
}
/* 🔥 SEPARATOR STYLE */
section[data-testid="stSidebar"] hr {
    border-color: #374151;
}


 /* 🔥 KEEP CONTENT ABOVE */
 /*.stApp > div {
    position: relative;
    z-index: 1;}*/
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
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from datetime import datetime
import time
import re
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

# ================================================================
# ── EMOTION ICON HELPER ──────────────────────────────────────────
# ================================================================
EMOTION_ICONS = {
    "Happy":    "😊",
    "Sad":      "😢",
    "Angry":    "😠",
    "Fear":     "😨",
    "Surprise": "😲",
    "Disgust":  "🤢",
    "Neutral":  "😐",
    "admiration":"🤩", "amusement":"😄", "anger":"😠",
    "annoyance":"😤", "approval":"👍", "caring":"🤗",
    "confusion":"😕", "curiosity":"🧐", "desire":"😍",
    "disappointment":"😞", "disapproval":"👎",
    "disgust":"🤢", "embarrassment":"😳", "excitement":"🤩",
    "fear":"😨", "gratitude":"🙏", "grief":"😭",
    "joy":"😂", "love":"❤️", "nervousness":"😰",
    "optimism":"🌟", "pride":"😌", "realization":"💡",
    "relief":"😮‍💨", "remorse":"😔", "sadness":"😢",
    "surprise":"😲", "neutral":"😐",
}

EMOTION_COLORS = {
    "Happy":    "#22c55e",
    "Sad":      "#3b82f6",
    "Angry":    "#ef4444",
    "Fear":     "#f59e0b",
    "Surprise": "#a855f7",
    "Disgust":  "#f97316",
    "Neutral":  "#9ca3af",
}

def emotion_color(e):
    return EMOTION_COLORS.get(e, "#00ffcc")

def emotion_icon(e):
    return EMOTION_ICONS.get(e, "🎭")

# ================================================================
# ── HOME ────────────────────────────────────────────────────────
# ================================================================
if page == "Home":
    st.title("🧠 Emotion AI Dashboard")

    col1, col2, col3 = st.columns(3)

    total = len(st.session_state.history)
    last  = st.session_state.history["Emotion"].iloc[-1] if total > 0 else "None"

    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>🔢 Total Predictions</h3>
            <h2 style="color:#00ffcc; font-size:38px; margin-top:10px;">{total}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        icon = emotion_icon(last)
        st.markdown(f"""
        <div class="card">
            <h3>🎭 Last Emotion</h3>
            <h2 style="color:#00ffcc; font-size:32px; margin-top:10px;">{icon} {last}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <h3>⚡ Status</h3>
            <h2 style="color:#22c55e; font-size:32px; margin-top:10px;">● Active</h2>
        </div>
        """, unsafe_allow_html=True)

    # ── How-to info cards ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card" style="text-align:left;">
            <h3>😊 Face Detection</h3>
            <p style="color:#d1d5db; margin-top:8px;">Upload any image containing a face.
            The AI will detect facial expressions and predict the emotion in real-time.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card" style="text-align:left;">
            <h3>💬 Text Detection</h3>
            <p style="color:#d1d5db; margin-top:8px;">Type any sentence or upload an image
            with text. The BERT model analyses sentiment and classifies the emotion.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card" style="text-align:left;">
            <h3>🎥 Live Detection</h3>
            <p style="color:#d1d5db; margin-top:8px;">Start your webcam for continuous
            real-time emotion tracking with smooth frame-by-frame analysis.</p>
        </div>
        """, unsafe_allow_html=True)


# ================================================================
# ── CAMERA DETECTION ────────────────────────────────────────────
# ================================================================
elif page == "Camera Detection":
    st.title("📷 Camera Emotion Detection")

    # ── control card ────────────────────────────────────────────
    st.markdown('<div class="card" style="text-align:left; margin-bottom:18px;">', unsafe_allow_html=True)

    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Start Camera"):
            st.session_state.camera_on = True
    with col_b:
        if st.button("⛔ Stop Camera"):
            st.session_state.camera_on = False

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.camera_on:
        # ── capture card ────────────────────────────────────────
        st.markdown('<div class="card" style="text-align:left;">', unsafe_allow_html=True)
        camera_image = st.camera_input("📸 Take a photo")
        st.markdown('</div>', unsafe_allow_html=True)

        if camera_image:
            image = Image.open(camera_image)
            img   = np.array(image)

            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            detected_label      = None
            detected_confidence = None

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
                face = face / 255.0
                face = np.reshape(face, (1, 48, 48, 3))

                pred               = face_model.predict(face, verbose=0)
                detected_label     = emotion_labels[np.argmax(pred)]
                detected_confidence = np.max(pred)

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img,
                            f"{detected_label} ({detected_confidence:.2f})",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # ── result cards ────────────────────────────────────
            r1, r2, r3 = st.columns(3)

            if detected_label:
                save_prediction("Camera", detected_label)
                color = emotion_color(detected_label)
                icon  = emotion_icon(detected_label)

                with r1:
                    st.markdown(f"""
                    <div class="card">
                        <h3>🎭 Detected Emotion</h3>
                        <h2 style="color:{color}; font-size:30px; margin-top:10px;">{icon} {detected_label}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with r2:
                    pct = round(float(detected_confidence) * 100, 1)
                    st.markdown(f"""
                    <div class="card">
                        <h3>📊 Confidence</h3>
                        <h2 style="color:#00ffcc; font-size:30px; margin-top:10px;">{pct}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with r3:
                    st.markdown(f"""
                    <div class="card">
                        <h3>👤 Faces Found</h3>
                        <h2 style="color:#a855f7; font-size:30px; margin-top:10px;">{len(faces)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card">
                    <h3 style="color:#ef4444;">⚠️ No face detected — please try again</h3>
                </div>
                """, unsafe_allow_html=True)

            # ── image card ──────────────────────────────────────
            st.markdown('<div class="card" style="margin-top:18px; text-align:left;">', unsafe_allow_html=True)
            st.image(img, caption="📷 Captured Frame", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ================================================================
# ── FACE DETECTION ──────────────────────────────────────────────
# ================================================================
elif page == "Face Detection":
    st.title("😊 Face Emotion Detection")

    # ── upload card ─────────────────────────────────────────────
    st.markdown('<div class="card" style="text-align:left; margin-bottom:18px;">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg","jpeg","png","webp"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        img   = np.array(image)

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_label      = None
        detected_confidence = None

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 3))

            pred               = face_model.predict(face, verbose=0)
            detected_label     = emotion_labels[np.argmax(pred)]
            detected_confidence = np.max(pred)

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img,
                        f"{detected_label} ({detected_confidence:.2f})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ── result cards ────────────────────────────────────────
        r1, r2, r3 = st.columns(3)

        if detected_label:
            save_prediction("Face", detected_label)
            color = emotion_color(detected_label)
            icon  = emotion_icon(detected_label)

            with r1:
                st.markdown(f"""
                <div class="card">
                    <h3>🎭 Detected Emotion</h3>
                    <h2 style="color:{color}; font-size:30px; margin-top:10px;">{icon} {detected_label}</h2>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                pct = round(float(detected_confidence) * 100, 1)
                st.markdown(f"""
                <div class="card">
                    <h3>📊 Confidence</h3>
                    <h2 style="color:#00ffcc; font-size:30px; margin-top:10px;">{pct}%</h2>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                st.markdown(f"""
                <div class="card">
                    <h3>👤 Faces Found</h3>
                    <h2 style="color:#a855f7; font-size:30px; margin-top:10px;">{len(faces)}</h2>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
                <h3 style="color:#ef4444;">⚠️ No face detected in the uploaded image</h3>
            </div>
            """, unsafe_allow_html=True)

        # ── image preview card ───────────────────────────────────
        st.markdown('<div class="card" style="margin-top:18px; text-align:left;">', unsafe_allow_html=True)
        st.image(img, caption="🖼️ Processed Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ================================================================
# ── TEXT DETECTION ──────────────────────────────────────────────
# ================================================================
elif page == "Text Detection":
    st.title("💬 Text Emotion Detection")

    # ── input card ──────────────────────────────────────────────
    st.markdown('<div class="card" style="text-align:left; margin-bottom:18px;">', unsafe_allow_html=True)
    text = st.text_input("✍️ Enter text to analyse")
    st.markdown('</div>', unsafe_allow_html=True)

    if text:
        result     = emotion_classifier(text)
        pred       = result[0]['label']
        confidence = result[0]['score']
        color      = emotion_color(pred)
        icon       = emotion_icon(pred)
        pct        = round(confidence * 100, 1)

        # ── result cards ────────────────────────────────────────
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
            <div class="card">
                <h3>🎭 Detected Emotion</h3>
                <h2 style="color:{color}; font-size:30px; margin-top:10px;">{icon} {pred}</h2>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="card">
                <h3>📊 Confidence</h3>
                <h2 style="color:#00ffcc; font-size:30px; margin-top:10px;">{pct}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="card">
                <h3>🔤 Input Words</h3>
                <h2 style="color:#a855f7; font-size:30px; margin-top:10px;">{len(text.split())}</h2>
            </div>
            """, unsafe_allow_html=True)

        save_prediction("Text", pred)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── image upload card ───────────────────────────────────────
    st.markdown('<div class="card" style="text-align:left; margin-bottom:18px;">', unsafe_allow_html=True)
    uploaded_img = st.file_uploader("🖼️ Upload Text Image", type=["png","jpg","jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_img:
        image          = Image.open(uploaded_img)
        extracted_text = pytesseract.image_to_string(image)

        # preview card
        st.markdown('<div class="card" style="text-align:left; margin-bottom:18px;">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f'<p style="color:#d1d5db; margin-top:12px;"><b>Extracted Text:</b><br>{extracted_text}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if extracted_text.strip():
            cleaned = clean_text(extracted_text)
            X       = vectorizer.transform([cleaned])
            pred    = text_model.predict(X)[0]
            color   = emotion_color(pred)
            icon    = emotion_icon(pred)

            # result cards
            ri1, ri2 = st.columns(2)
            with ri1:
                st.markdown(f"""
                <div class="card">
                    <h3>🎭 Detected Emotion</h3>
                    <h2 style="color:{color}; font-size:30px; margin-top:10px;">{icon} {pred}</h2>
                </div>
                """, unsafe_allow_html=True)

            with ri2:
                st.markdown(f"""
                <div class="card">
                    <h3>🔤 Words Extracted</h3>
                    <h2 style="color:#a855f7; font-size:30px; margin-top:10px;">{len(extracted_text.split())}</h2>
                </div>
                """, unsafe_allow_html=True)

            save_prediction("Image Text", pred)


# ================================================================
# ── LIVE DETECTION ──────────────────────────────────────────────
# ================================================================
elif page == "Live Detection":
    st.header("🎥 Live Emotion Detection")

    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    # ── control card ────────────────────────────────────────────
    st.markdown('<div class="card" style="text-align:left; margin-bottom:18px;">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Camera"):
            st.session_state.camera_on = True
    with col2:
        if st.button("⛔ Stop Camera"):
            st.session_state.camera_on = False
    st.markdown('</div>', unsafe_allow_html=True)

    # ── status card ─────────────────────────────────────────────
    status_color = "#22c55e" if st.session_state.camera_on else "#ef4444"
    status_text  = "● Running" if st.session_state.camera_on else "● Stopped"
    st.markdown(f"""
    <div class="card" style="margin-bottom:18px;">
        <h3>⚡ Camera Status</h3>
        <h2 style="color:{status_color}; font-size:28px; margin-top:10px;">{status_text}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── live feed card ──────────────────────────────────────────
    st.markdown('<div class="card" style="text-align:left;">', unsafe_allow_html=True)
    FRAME_WINDOW = st.image([])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── emotion display placeholder ─────────────────────────────
    emotion_placeholder = st.empty()

    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.markdown("""
            <div class="card">
                <h3 style="color:#ef4444;">❌ Camera not accessible</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            from collections import deque
            emotion_buffer = deque(maxlen=10)

            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    break

                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 3)

                current_emotion = None

                for (x, y, w, h) in faces[:1]:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (48, 48))
                    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
                    face = face / 255.0
                    face = np.reshape(face, (1, 48, 48, 3))

                    pred       = face_model(face, training=False).numpy()
                    lbl        = emotion_labels[np.argmax(pred)]
                    confidence = float(np.max(pred))

                    if confidence > 0.6:
                        emotion_buffer.append(lbl)

                    smooth_emotion    = max(set(emotion_buffer), key=emotion_buffer.count) if emotion_buffer else lbl
                    confidence_pct    = confidence * 100
                    current_emotion   = smooth_emotion

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame,
                                f"{smooth_emotion} ({confidence_pct:.1f}%)",
                                (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if smooth_emotion != st.session_state.last_saved_emotion:
                        save_prediction("Live", smooth_emotion)
                        st.session_state.last_saved_emotion = smooth_emotion

                FRAME_WINDOW.image(frame, channels="BGR")

                # live emotion card update
                if current_emotion:
                    color = emotion_color(current_emotion)
                    icon  = emotion_icon(current_emotion)
                    emotion_placeholder.markdown(f"""
                    <div class="card" style="margin-top:14px;">
                        <h3>🎭 Live Emotion</h3>
                        <h2 style="color:{color}; font-size:32px; margin-top:10px;">{icon} {current_emotion}</h2>
                    </div>
                    """, unsafe_allow_html=True)

            cap.release()


# ================================================================
# ── DASHBOARD ───────────────────────────────────────────────────
# ================================================================
elif page == "Dashboard":
    st.title("📊 Analytics Dashboard")

    df = st.session_state.history

    if len(df) == 0:
        st.markdown("""
        <div class="card">
            <h3>⚠️ No data yet — use any detection feature first!</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_display       = df.copy()
        df_display["Time"] = pd.to_datetime(df_display["Time"], format="%H:%M:%S")

        total = len(df)
        most  = df["Emotion"].value_counts().idxmax()
        last  = df["Emotion"].iloc[-1]

        # ── summary cards row ────────────────────────────────────
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown(f"""
            <div class="card">
                <h3>🔢 Total Predictions</h3>
                <h2 style="color:#00ffcc; font-size:34px; margin-top:10px;">{total}</h2>
            </div>
            """, unsafe_allow_html=True)

        with s2:
            color = emotion_color(most)
            icon  = emotion_icon(most)
            st.markdown(f"""
            <div class="card">
                <h3>🏆 Most Frequent</h3>
                <h2 style="color:{color}; font-size:28px; margin-top:10px;">{icon} {most}</h2>
            </div>
            """, unsafe_allow_html=True)

        with s3:
            color2 = emotion_color(last)
            icon2  = emotion_icon(last)
            st.markdown(f"""
            <div class="card">
                <h3>🕐 Last Emotion</h3>
                <h2 style="color:{color2}; font-size:28px; margin-top:10px;">{icon2} {last}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── charts row ───────────────────────────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown('<div class="card" style="text-align:left;">', unsafe_allow_html=True)
            st.subheader("📈 Emotion Timeline")
            timeline = df_display.groupby("Time").count()["Emotion"]
            st.line_chart(timeline)
            st.markdown('</div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div class="card" style="text-align:left;">', unsafe_allow_html=True)
            st.subheader("📊 Emotion Distribution")
            st.bar_chart(df["Emotion"].value_counts())
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── pie chart card ───────────────────────────────────────
        st.markdown('<div class="card" style="text-align:left;">', unsafe_allow_html=True)
        st.subheader("🥧 Emotion Pie Chart")
        st.write(
            df["Emotion"].value_counts().plot.pie(autopct="%1.1f%%").figure
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── history table card ───────────────────────────────────
        st.markdown('<div class="card" style="text-align:left;">', unsafe_allow_html=True)
        st.subheader("📜 Full History")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv, "emotion_history.csv")
        st.markdown('</div>', unsafe_allow_html=True)