🎶 Moodify

AI-powered music player that detects your facial emotion and plays songs to match your mood.

📌 Overview

Moodify is an interactive Streamlit web app that uses your webcam to capture facial expressions, detect emotions with a trained CNN model, and automatically play songs that fit your mood.

Currently, it supports Happy, Sad, Neutral, and Surprise emotions.

✨ Features

🎥 Captures real-time facial expressions from webcam

🤖 Detects dominant emotion using a CNN model (emotion_cnn.h5)

🎶 Plays mood-based songs (local MP3 files)

🔄 Option to re-detect emotion or change song

⏹️ Stop music anytime

🛠️ Tech Stack

Python

Streamlit
 – Web app framework

OpenCV
 – Webcam and face detection

TensorFlow / Keras
 – Deep learning model

NumPy
 – Numerical processing

📂 Project Structure
moodify/
├── train.py                #training the model
├── emotion_cnn.h5          # Pre-trained emotion detection model
├── streamlit.py                  # Main Streamlit app
├── songs/                  # Local folder containing mp3 files
│   ├── happySong1,happySong2......
│   ├── sad1,sad2.....
│   ├── neutral1,neutral2....
│   └── surprise1,surprise2....
└── README.md

⚙️ Installation

Clone the repository

git clone https://github.com/your-username/moodify.git
cd moodify


Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt



▶️ Usage

Run the Streamlit app:

streamlit run app.py


Click ▶ Start Detection → webcam captures for 5 seconds

The app detects your dominant emotion

A song matching your mood will play automatically

You can:

🔄 Re-detect emotion

⏭ Change song

⏹ Stop song

📊 Example
🎵 Moodify started!
📸 Capturing your emotion for 5 seconds...
😀 Final Detected Emotion: Happy
🎶 Playing: Santhoshakke Haadu Santhoshakke.mp3

🚀 Future Improvements

Add support for more emotions (Angry, Fear, Disgust)

Integrate with Spotify API / YouTube API for online music

Improve accuracy with a fine-tuned model

Deploy on cloud (Streamlit Cloud, Heroku, etc.)

🙌 Acknowledgments

Streamlit

TensorFlow

OpenCV

Kaggle FER Dataset
 (for emotion training)




