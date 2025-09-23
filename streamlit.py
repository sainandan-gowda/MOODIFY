import streamlit as st
import cv2
import numpy as np
import time
from collections import Counter
from tensorflow.keras.models import load_model
import random
import os
from PIL import Image


try:
    model = load_model("emotion_cnn.h5")
    st.success("✅ Emotion model loaded successfully!")
except:
    st.error("❌ Could not load emotion_cnn.h5 model. Please check the file path.")
    st.info("Using demo mode with random emotion detection.")

# Emotion labels and mapping
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
label_map = {
    'Angry': 'Sad', 
    'Disgust': 'Surprise',
    'Fear': 'Surprise',
    'Happy': 'Happy', 
    'Neutral': 'Neutral',
    'Sad': 'Sad', 
    'Surprise': 'Surprise'
} 



songs = {
    "Happy": [
        {"title": "Santhoshakke Haadu", "artist": "Geetha", "path": "E:\emotion\Santhoshakke Haadu Santhoshakke - Geetha - HD Video Song  Shankarnag  Ilayaraja  S.P.B.mp3", "color": "#FFD700"},
        {"title": "All OK", "artist": "Kannada Hits", "path": "E:\emotion\ALL OK  Happy Video   New Kannada Song.mp3", "color": "#FF6B6B"}
    ],
    "Sad": [
        {"title": "Paro", "artist": "Aditya Rikhari", "path": "E:\emotion\Paro - @adityarikhari  UNPLG'd.mp3", "color": "#4ECDC4"},
        {"title": "Kabira", "artist": "Arijit Singh", "path": "E:\emotion\Kabira Full Lyrics Song Yeh Jawaani Hai Deewani Ranbir Kapoor, Deepika Padukone.mp3", "color": "#45B7D1"}
    ],
    "Neutral": [
        {"title": "Baana Daariyalli", "artist": "Puneeth Rajkumar", "path": "E:\emotion\Baana Daariyalli Surya Jaari Hoda Song - Puneeth Rajkumar  Reprise Puneeth Rajkumar Songs  Kannada.mp3", "color": "#96CEB4"},
        {"title": "Sahiba", "artist": "T-Series", "path": "E:\emotion\Sahiba (Official Music Video) ： Aditya Rikhari, Ankita Chhetri ｜ T-Series.webm", "color": "#FFEAA7"}
    ],
    "Surprise": [
        {"title": "DANKS ANTHEM", "artist": "Anurag Kulkarni", "path": "E:\emotion\DANKS ANTHEM - Video Song  Su From So  Anurag Kulkarni  Sumedh K  Raj B Shetty  J P Thuminad.mp3", "color": "#DDA0DD"},
        {"title": "Aaluma Doluma", "artist": "Ajith Kumar", "path": "E:\emotion\Vedalam - Aaluma Doluma Video with Lyrics  Ajith Kumar  Anirudh.mp3", "color": "#98D8C8"}
    ],
}


custom_css = """<style>
/* Animated gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    min-height: 100vh;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glass morphism sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
}

/* Modern card design */
.emotion-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}

.emotion-card:hover {
    transform: translateY(-5px);
}

/* Webcam feed styling */
.webcam-container {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 15px;
    padding: 15px;
    margin: 15px 0;
    text-align: center;
}

/* Animated buttons */
.stButton>button {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 30px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    margin: 5px;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

/* Music player styling */
.music-player {
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    margin: 20px 0;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Progress bar animation */
.stProgress > div > div {
    background: linear-gradient(90deg, #ff6b6b, #ee5a24);
    border-radius: 10px;
}

/* Floating animation */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

.floating {
    animation: float 3s ease-in-out infinite;
}


@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.pulse {
    animation: pulse 2s infinite;
}

/* Detection animation */
@keyframes detecting {
    0% { opacity: 0.5; }
    50% { opacity: 1; }
    100% { opacity: 0.5; }
}

.detecting {
    animation: detecting 1.5s infinite;
}


.emotion-icon {
    font-size: 3em;
    margin: 10px;
    transition: all 0.3s ease;
}

.emotion-icon:hover {
    transform: scale(1.2) rotate(10deg);
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state
if 'final_emotion' not in st.session_state:
    st.session_state.final_emotion = None
if 'current_song_index' not in st.session_state:
    st.session_state.current_song_index = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'volume' not in st.session_state:
    st.session_state.volume = 80
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

def detect_emotion_webcam():
    """Function to capture webcam feed and detect emotions"""
    st.session_state.webcam_active = True
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check your camera permissions.")
        st.session_state.webcam_active = False
        return None
    
   
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    st.info("🎥 Webcam activated! Looking for faces...")
    st.warning("⚠️ Make sure your face is clearly visible in the camera.")
    
  
    webcam_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    collected_emotions = []
    start_time = time.time()
    detection_duration = 5  # seconds
    
    try:
        while time.time() - start_time < detection_duration:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            detected_emotion = "No Face"
            emotion_color = (0, 0, 255)  # Red for no face
            
            for (x, y, w, h) in faces:
                # Extract face region
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)
                
                # Predict emotion
                try:
                    preds = model.predict(roi_gray, verbose=0)
                    emotion_idx = np.argmax(preds[0])
                    raw_emotion = emotion_labels[emotion_idx]
                    detected_emotion = label_map[raw_emotion]
                    collected_emotions.append(detected_emotion)
                    
                    # Set color based on emotion
                    emotion_colors = {
                        "Happy": (0, 255, 0),      # Green
                        "Sad": (255, 0, 0),        # Blue
                        "Neutral": (255, 255, 0),  # Cyan
                        "Surprise": (0, 165, 255)  # Orange
                    }
                    emotion_color = emotion_colors.get(detected_emotion, (255, 255, 255))
                    
                except Exception as e:
                    detected_emotion = "Error"
                    emotion_color = (0, 0, 255)
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_color, 2)
                cv2.putText(frame, detected_emotion, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display webcam feed
            webcam_placeholder.image(frame_rgb, channels="RGB", 
                                   caption=f"Live Webcam Feed - {detected_emotion}", 
                                   use_column_width=True)
            
            # Update progress
            elapsed_time = time.time() - start_time
            progress = min(int((elapsed_time / detection_duration) * 100), 100)
            progress_bar.progress(progress)
            
            status_placeholder.markdown(f"""
            <div class="detecting" style="text-align: center; padding: 10px; background: rgba(255,255,255,0.2); border-radius: 10px;">
                <h3>🔍 Detecting Emotions... {progress}%</h3>
                <p>Current detection: <strong>{detected_emotion}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(0.1)  # Small delay to control frame rate
            
    except Exception as e:
        st.error(f"Error during emotion detection: {str(e)}")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        st.session_state.webcam_active = False
    
    # Determine final emotion
    if collected_emotions:
        emotion_counter = Counter(collected_emotions)
        final_emotion = emotion_counter.most_common(1)[0][0]
        confidence = emotion_counter[final_emotion] / len(collected_emotions)
        
        st.success(f"✅ Emotion detected: **{final_emotion}** (Confidence: {confidence:.2%})")
        return final_emotion
    else:
        st.warning("⚠️ No faces detected. Using random emotion selection.")
        return random.choice(["Happy", "Sad", "Neutral", "Surprise"])

# Header with animated title
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 class="floating">🎵 MOODIFY </h1>
    <p style="font-size: 1.2em;">a musical mind-reader—it figures out your mood and serves the perfect tunes before you even say “sad playlist.”</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for user profile and history
with st.sidebar:
    st.markdown("### 👤 User Profile")
    st.image("https://via.placeholder.com/150/667eea/ffffff?text=USER", width=150)
    st.write("**Welcome, Music Lover!**")
    
    st.markdown("### 📊 Emotion History")
    if st.session_state.detection_history:
        for i, (emotion, timestamp) in enumerate(st.session_state.detection_history[-5:]):
            st.write(f"• {emotion} - {timestamp}")
    else:
        st.write("No detection history yet")
    
    st.markdown("### ⚙️ Settings")
    st.session_state.volume = st.slider("Volume", 0, 100, st.session_state.volume)
    auto_play = st.checkbox("Auto-play next song", value=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Emotion detection section
    st.markdown("### 🎭 Emotion Detection")
    
    # Webcam feed section
    st.markdown("#### 📸 Webcam Detection")
    st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
    
    if st.button("🎬 Start Webcam Detection", key="start_webcam"):
        with st.spinner("Initializing webcam..."):
            final_emotion = detect_emotion_webcam()
            if final_emotion:
                st.session_state.final_emotion = final_emotion
                st.session_state.current_song_index = 0
                st.session_state.detection_history.append(
                    (final_emotion, time.strftime("%H:%M:%S"))
                )
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick emotion selection
    st.markdown("#### ⚡ Quick Emotion Select")
    emotion_cols = st.columns(4)
    quick_emotions = ["Happy", "Sad", "Neutral", "Surprise"]
    emotion_icons = ["😊", "😢", "😐", "😲"]
    
    for i, (emotion, icon) in enumerate(zip(quick_emotions, emotion_icons)):
        with emotion_cols[i]:
            if st.button(f"{icon} {emotion}", key=f"quick_{emotion}"):
                st.session_state.final_emotion = emotion
                st.session_state.current_song_index = 0
                st.session_state.detection_history.append(
                    (emotion, time.strftime("%H:%M:%S"))
                )
                st.rerun()

with col2:
    # Current emotion display
    if st.session_state.final_emotion:
        emotion_color = {
            "Happy": "#FFD700", "Sad": "#4ECDC4", 
            "Neutral": "#96CEB4", "Surprise": "#DDA0DD"
        }
        color = emotion_color.get(st.session_state.final_emotion, "#FFFFFF")
        
        st.markdown(f"""
        <div class="emotion-card" style="border-left: 5px solid {color};">
            <h3>Current Mood</h3>
            <div style="text-align: center;">
                <div class="emotion-icon pulse" style="font-size: 4em;">
                    {['😊', '😢', '😐', '😲'][quick_emotions.index(st.session_state.final_emotion)] if st.session_state.final_emotion in quick_emotions else '😊'}
                </div>
                <h2 style="color: {color}; margin: 10px 0;">{st.session_state.final_emotion}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Additional control buttons
col1, col2, col3 = st.columns(3)

with col2:
    if st.button("🎲 Random Mood", key="random_mood"):
        random_emotion = random.choice(quick_emotions)
        st.session_state.final_emotion = random_emotion
        st.session_state.current_song_index = 0
        st.session_state.detection_history.append(
            (random_emotion, time.strftime("%H:%M:%S"))
        )
        st.rerun()

# Music player section
if st.session_state.final_emotion:
    st.markdown("---")
    st.markdown("### 🎵 Music Player")
    
    current_emotion = st.session_state.final_emotion
    song_list = songs[current_emotion]
    current_song = song_list[st.session_state.current_song_index]
    
    # Music player card
    st.markdown(f"""
    <div class="music-player">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="flex: 1;">
                <h3 style="color: {current_song['color']}; margin: 0;">{current_song['title']}</h3>
                <p style="margin: 5px 0; opacity: 0.8;">{current_song['artist']}</p>
                <p style="margin: 0; font-size: 0.9em; opacity: 0.6;">Mood: {current_emotion}</p>
            </div>
            <div style="font-size: 3em; margin-left: 20px;">
                {['😊', '😢', '😐', '😲'][quick_emotions.index(current_emotion)] if current_emotion in quick_emotions else '😊'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Audio player
    audio_placeholder = st.empty()
    audio_placeholder.audio(current_song["path"], format="audio/mp3")
    
    # Player controls
    control_col1, control_col2, control_col3, control_col4 = st.columns([1, 1, 1, 2])
    
    with control_col1:
        if st.button("⏮ Previous"):
            st.session_state.current_song_index = (st.session_state.current_song_index - 1) % len(song_list)
            st.rerun()
    
    with control_col2:
        play_pause_text = "⏸ Pause" if st.session_state.is_playing else "▶ Play"
        if st.button(play_pause_text):
            st.session_state.is_playing = not st.session_state.is_playing
    
    with control_col3:
        if st.button("⏭ Next"):
            st.session_state.current_song_index = (st.session_state.current_song_index + 1) % len(song_list)
            st.rerun()
    
    with control_col4:
        st.write(f"Track: {st.session_state.current_song_index + 1} of {len(song_list)}")
    
    # Progress bar simulation
    if st.session_state.is_playing:
        progress = st.progress(st.session_state.progress)
        if st.session_state.progress < 100:
            st.session_state.progress += 1
        else:
            st.session_state.progress = 0
            if auto_play:
                st.session_state.current_song_index = (st.session_state.current_song_index + 1) % len(song_list)
                st.rerun()
    
    # Song recommendations for current emotion
    st.markdown("### 💫 Recommended Playlist")
    rec_cols = st.columns(3)
    for i, song in enumerate(song_list):
        with rec_cols[i % 3]:
            st.markdown(f"""
            <div class="emotion-card">
                <div style="color: {song['color']}; font-weight: bold;">{song['title']}</div>
                <div style="font-size: 0.9em; opacity: 0.8;">{song['artist']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Play", key=f"play_{i}"):
                st.session_state.current_song_index = i
                st.session_state.is_playing = True
                st.rerun()

# Statistics section
st.markdown("---")
st.markdown("### 📈 Your Music Statistics")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

emotion_counts = Counter([h[0] for h in st.session_state.detection_history])

with stat_col1:
    st.markdown(f"""
    <div class="emotion-card">
        <h3>🎵 Total Sessions</h3>
        <h2>{len(st.session_state.detection_history)}</h2>
    </div>
    """, unsafe_allow_html=True)

with stat_col2:
    st.markdown(f"""
    <div class="emotion-card">
        <h3>😊 Happy Moments</h3>
        <h2>{emotion_counts.get('Happy', 0)}</h2>
    </div>
    """, unsafe_allow_html=True)

with stat_col3:
    st.markdown(f"""
    <div class="emotion-card">
        <h3>😢 Sad Moments</h3>
        <h2>{emotion_counts.get('Sad', 0)}</h2>
    </div>
    """, unsafe_allow_html=True)

with stat_col4:
    st.markdown(f"""
    <div class="emotion-card">
        <h3>⚡ Surprise Moments</h3>
        <h2>{emotion_counts.get('Surprise', 0)}</h2>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; opacity: 0.7;">
    <p>🎶 Moodify  - Where emotions meet music • Made with ❤️</p>
    <p><small>Note: Using online audio samples. Replace with your local MP3 files.</small></p>
</div>
""", unsafe_allow_html=True)
