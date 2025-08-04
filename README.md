# 🎙️ Real-Time Voice Transcription App

A lightweight real-time voice transcription system powered by WebSocket audio streaming and Faster-Whisper, Silero VAD for accurate, fast speech-to-text on GPU and CPU.

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

## ⚠️ Requirements for GPU setup

- cuDNN version **9.1 or higher**

**Step 1: Add NVIDIA repository (if not already added)**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
```

**Step 2: Install cuDNN 9.1**
```bash
sudo apt install libcudnn9-dev libcudnn9-cuda-12
```

**Alternative method using direct download:**

If the apt method doesn't work, we'll download directly from NVIDIA:

1. Go to https://developer.nvidia.com/cudnn (requires free NVIDIA account)
2. Download cuDNN 9.1.x for CUDA 12.x
3. Extract and install manually


**Step 3: Verify installation**
```bash
find /usr -name "*cudnn*" 2>/dev/null | grep "\.so\.9"
```

**Step 4: Set LD_LIBRARY_PATH temporarily**
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

**Step 5 (Optional): Make LD_LIBRARY_PATH permanent**
Add this line to your `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
---

## 👨‍💻 About Me

I'm an engineer and curious learner passionate about deep learning, AI/ML application development, data pipelines, and automation.

📧 Email: *sagardekaofficial@gmail.com*
