import streamlit as st
import cv2
import numpy as np
import time
from collections import Counter
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_cnn.h5")

# Original 7-class labels from training
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Map 7 → 4 emotions
label_map = {
    'Angry': 'Sad',
    'Disgust': 'Surprise',
    'Fear': 'Surprise',
    'Happy': 'Happy',
    'Neutral': 'Neutral',
    'Sad': 'Sad',
    'Surprise': 'Surprise'
} 

# Songs(mp3 paths)
songs = {
    "Happy": [
        r"E:\emotion\Santhoshakke Haadu Santhoshakke - Geetha - HD Video Song  Shankarnag  Ilayaraja  S.P.B.mp3",
        r"ALL OK  Happy Video   New Kannada Song.mp3"
    ],
    "Sad": [
        r"E:\emotion\Paro - @adityarikhari  UNPLG'd.mp3",
        r"E:\emotion\Kabira Full Lyrics Song Yeh Jawaani Hai Deewani Ranbir Kapoor, Deepika Padukone.mp3"
    ],
    "Neutral": [
        r"E:\emotion\Baana Daariyalli Surya Jaari Hoda Song - Puneeth Rajkumar  Reprise Puneeth Rajkumar Songs  Kannada.mp3",
        r"E:\emotion\Sahiba (Official Music Video) ： Aditya Rikhari, Ankita Chhetri ｜ T-Series.webm"
    ],
    "Surprise": [
        r"E:\emotion\DANKS ANTHEM - Video Song  Su From So  Anurag Kulkarni  Sumedh K  Raj B Shetty  J P Thuminad.mp3",
        r"E:\emotion\Vedalam - Aaluma Doluma Video with Lyrics  Ajith Kumar  Anirudh.mp3"
    ],
}

# Streamlit UI
st.title("🎵 Emotion Based Music Player")
st.write("Webcam captures for 5 seconds → Detects dominant emotion → Plays a song")

video_placeholder = st.empty()
emotion_placeholder = st.empty()
audio_placeholder = st.empty()

# Store session state
if 'final_emotion' not in st.session_state:
    st.session_state.final_emotion = None
if 'current_song_index' not in st.session_state:
    st.session_state.current_song_index = 0

def detect_emotion():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    st.write("📸 Capturing your emotion for 5 seconds...")
    start_time = time.time()
    collected_emotions = []

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Could not access webcam.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_emotion = "Neutral"  # default
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float32") / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            preds = model.predict(roi_gray)
            raw_emotion = emotion_labels[np.argmax(preds[0])]   # 7-class output
            detected_emotion = label_map[raw_emotion]           # map → 4 classes
            collected_emotions.append(detected_emotion)

            # Draw rectangle + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

    cap.release()

    if collected_emotions:
        final_emotion = Counter(collected_emotions).most_common(1)[0][0]
        st.session_state.final_emotion = final_emotion
        st.session_state.current_song_index = 0
        emotion_placeholder.markdown(f"### 😀 Final Detected Emotion: **{final_emotion}**")
        play_song(final_emotion)
    else:
        st.warning("No face detected in the 5-second window.")

def play_song(emotion):
    if emotion in songs:
        song_list = songs[emotion]
        index = st.session_state.current_song_index
        current_song = song_list[index]
        audio_placeholder.audio(current_song, format="audio/mp3")
        st.write(f"🎵 Playing song {index+1} of {len(song_list)} for emotion: {emotion}")

# Buttons
col1, col2, col3 = st.columns([1,1,1])

with col1:
    if st.button("▶ Start Detection"):
        detect_emotion()

with col2:
    if st.button("🔄 Detect Again"):
        detect_emotion()

with col3:
    if st.session_state.final_emotion:
        if st.button("⏭ Change Song"):
            song_list = songs[st.session_state.final_emotion]
            st.session_state.current_song_index = (st.session_state.current_song_index + 1) % len(song_list)
            play_song(st.session_state.final_emotion)

# Stop song
if st.session_state.final_emotion and st.button("⏹ Stop Song"):
    audio_placeholder.empty()
    st.write("⏹️ Song stopped.")
