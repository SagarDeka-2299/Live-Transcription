# 🎙️ Real-Time Voice Transcription App

A lightweight real-time voice transcription system powered by WebSocket audio streaming and Faster-Whisper for accurate, fast speech-to-text on GPU.

---

## ✅ Features

- Live audio streaming from browser
- Filters silence and background noise
- Segments speech naturally at pauses
- Transcribes in real-time using Whisper (via Faster-Whisper)
- Supports both CPU and GPU (recommended)

---

## 🖼️ Live App Screenshots  

<img width="1387" height="794" alt="Screenshot 2025-07-23 at 1 32 04 PM" src="https://github.com/user-attachments/assets/f8868990-7d19-4784-9839-4a234f56d443" />
<img width="1358" height="786" alt="Screenshot 2025-07-23 at 1 32 22 PM" src="https://github.com/user-attachments/assets/8250170a-bdb7-484e-a7d0-368c4c34f75e" />


---

## 🧠 Architecture Overview

Browser captures mic input and streams audio chunks over WebSocket.  
Server buffers and segments audio at silent points, then uses Faster-Whisper to transcribe speech.  
Optimized for GPU (CUDA + cuDNN) but also supports CPU fallback.

<img width="1024" height="1024" alt="Untitled Design 1024x1024" src="https://github.com/user-attachments/assets/a0759297-cf56-4dea-9ef7-0416b2ca8dd0" />


---

## ⚙️ Setup

1. Clone the repository  
2. Install requirements: `pip install -r requirements.txt`  

---

## ⚠️ Requirements

- cuDNN version **9.1 or higher**
- CUDA-enabled GPU recommended for real-time performance

---

## 👨‍💻 About Me

I'm an engineer and curious learner passionate about deep learning, AI/ML application development, data pipelines, and automation.

📧 Email: *[your-gmail]*  
🔗 LinkedIn: *[your-linkedin]*  
