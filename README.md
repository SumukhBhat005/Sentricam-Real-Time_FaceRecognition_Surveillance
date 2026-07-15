<div align="center">

# 👁️ SentriCam

### AI-Powered Real-Time Biometric Surveillance System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FaceNet-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-Facial_Analysis-00B4D8?style=for-the-badge)](https://github.com/serengil/deepface)
[![Flask](https://img.shields.io/badge/Flask-Web_Server-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

---

*A production-grade, privacy-first biometric surveillance platform that performs real-time face detection, recognition, and identity tracking at 30fps — entirely on-device with zero cloud dependencies. Built with a multi-threaded architecture featuring decoupled camera capture, AI inference, and MJPEG streaming pipelines.*

[View Demo »](#-demo) · [Architecture »](#-system-architecture) · [Getting Started »](#-getting-started)

</div>

---

## 📋 Table of Contents

- [Why SentriCam?](#-why-sentricam)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [System Architecture](#-system-architecture)
- [Technical Deep Dive](#-technical-deep-dive)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Performance Benchmarks](#-performance-benchmarks)
- [Privacy & Security](#-privacy--security)
- [Future Roadmap](#-future-roadmap)
- [License](#-license)

---

## 💡 Why SentriCam?

Most facial recognition systems rely on cloud APIs (AWS Rekognition, Azure Face, Google Vision) — introducing **latency**, **privacy risks**, and **recurring costs**. SentriCam was built to solve all three:

| Challenge | Cloud Solutions | SentriCam |
|:---|:---:|:---:|
| **Latency** | 200-500ms API roundtrip | **< 33ms** (local inference) |
| **Privacy** | Biometric data sent to third-party servers | **100% on-device** processing |
| **Cost** | $1-4 per 1,000 API calls | **$0** — fully self-hosted |
| **Offline Capability** | ❌ Requires internet | ✅ Works completely offline |
| **Data Sovereignty** | Data stored on vendor's cloud | ✅ Your data never leaves your machine |

---

## ✨ Key Features

### 🎥 Real-Time Video Pipeline
- **30fps MJPEG Streaming** — Hardware-accelerated camera capture with browser-native rendering
- **Dual-Cascade Face Detection** — Combines frontal and profile Haar classifiers for robust multi-angle detection
- **EMA-Smoothed Tracking** — Exponential Moving Average filter eliminates bounding box jitter for silky-smooth UI overlays
- **Track Persistence** — Detected faces maintain identity for ~0.5s even during brief occlusions (15-frame TTL buffer)

### 🧠 AI-Powered Recognition
- **FaceNet Embeddings** — Google's FaceNet model via DeepFace generates 128-dimensional facial embeddings
- **Cosine Similarity Matching** — Custom distance function compares live embeddings against enrolled profiles
- **Multi-Shot Enrollment** — Register multiple photos per person for higher accuracy across lighting conditions and angles
- **Configurable Threshold** — Adjustable cosine distance threshold (default: 0.40) to tune precision vs. recall

### 📊 Intelligent Dashboard
- **Live Detection Feed** — Color-coded bounding boxes: 🟢 Known | 🔴 Unknown | 🟠 Scanning
- **Real-Time Statistics** — Today's detections, unknown alerts, enrolled users, last-seen tracking
- **Activity Log** — Searchable, time-stamped event log of all recognized and unknown encounters
- **User Management** — Enroll, view, and remove identities directly from the web interface
- **Runtime Configuration** — Adjust detection threshold, alert cooldown, and stream quality without restarting

### 🔒 Privacy-First Design
- **Zero Cloud Dependencies** — No API keys, no external services, no data transmission
- **Local-Only Processing** — All AI inference runs on your CPU/GPU using TensorFlow
- **Self-Contained Database** — SQLite with serialized embeddings stored as binary blobs

---

## 🎬 Demo

### Dashboard Interface

```
┌─────────────────────────────────────────────────────────────────────┐
│  SentriCam Control Panel                              ⚙️  Settings │
├────────────────────────────────────┬────────────────────────────────┤
│                                    │  📊 Live Statistics            │
│    ┌──────────────────────────┐    │  ┌──────────┬──────────┐      │
│    │                          │    │  │ Detected │ Unknown  │      │
│    │   🎥 Live Camera Feed   │    │  │   127    │    3     │      │
│    │                          │    │  └──────────┴──────────┘      │
│    │  ┌────────┐  ┌────────┐ │    │  ┌──────────┬──────────┐      │
│    │  │ SUMUKH │  │UNKNOWN │ │    │  │ Enrolled │ Last Seen│      │
│    │  │  🟢    │  │  🔴    │ │    │  │    5     │ Sumukh   │      │
│    │  └────────┘  └────────┘ │    │  └──────────┴──────────┘      │
│    │                          │    │                                │
│    └──────────────────────────┘    │  👥 Enrolled Users             │
│                                    │  ├─ Sumukh (3 photos)          │
│    ▶ LIVE · 30fps · 1280×720       │  ├─ Alex (2 photos)            │
│                                    │  └─ Priya (5 photos)           │
├────────────────────────────────────┤                                │
│  📋 Activity Log                   │  ➕ Enroll New Person          │
│  16:42:15  Sumukh    [0.21]       │  ┌──────────────────────────┐  │
│  16:41:58  Unknown   [0.67]  ⚠️   │  │ Name: ________________  │  │
│  16:41:33  Sumukh    [0.19]       │  │      [ 📸 Capture ]     │  │
│  16:40:12  Alex      [0.28]       │  └──────────────────────────┘  │
└────────────────────────────────────┴────────────────────────────────┘
```

---

## 🏗️ System Architecture

SentriCam uses a **multi-threaded, event-driven architecture** with three decoupled pipelines running concurrently:

```
                          ┌─────────────────────────────┐
                          │      Web Browser Client      │
                          │   (Dashboard + MJPEG Feed)   │
                          └──────────┬──────────────────┘
                                     │ HTTP
                          ┌──────────▼──────────────────┐
                          │      Flask Web Server        │
                          │    (REST API + Routing)       │
                          └──────────┬──────────────────┘
                                     │
              ┌──────────────────────┼───────────────────────┐
              │                      │                       │
    ┌─────────▼──────────┐ ┌────────▼─────────┐  ┌─────────▼──────────┐
    │   Camera Thread     │ │  Engine Thread   │  │  MJPEG Generator   │
    │   (30fps Capture)   │ │  (AI Inference)  │  │  (Stream Output)   │
    │                     │ │                  │  │                    │
    │ • VideoCapture      │ │ • FaceNet Model  │  │ • JPEG Encoding    │
    │ • Haar Detection    │ │ • Cosine Match   │  │ • Multipart HTTP   │
    │ • EMA Smoothing     │ │ • ID Resolution  │  │ • Rate Limiting    │
    │ • Track Management  │ │ • Batch Process  │  │                    │
    └─────────┬──────────┘ └────────▲─────────┘  └─────────▲──────────┘
              │                     │                      │
              │   Face Crops        │   Results Dict       │  Annotated
              └─────────────────────┘                      │  Frames
              │                                            │
              └────────────────────────────────────────────┘
                        Shared Frame Buffer (Thread-Safe)

    ┌─────────────────────────────────────────────────────────────────┐
    │                    SQLite Database (WAL Mode)                    │
    │  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
    │  │   users      │  │  detection_logs  │  │  Pickled Binary   │  │
    │  │  (id, name,  │  │  (name, conf,    │  │  Embeddings       │  │
    │  │   embedding) │  │   timestamp)     │  │  (128-dim float)  │  │
    │  └─────────────┘  └──────────────────┘  └───────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘
```

### Thread Synchronization Model

| Thread | Frequency | Responsibility | Shared Resources |
|:---|:---:|:---|:---|
| **Camera Thread** | ~30 fps | Captures frames, runs Haar detection, applies EMA smoothing, sends face crops to engine | `frame_lock`, `camera_lock` |
| **Engine Thread** | ~2-5 fps | Runs FaceNet inference on cropped faces, performs cosine similarity matching | `engine.lock` |
| **MJPEG Generator** | ~30 fps | Reads annotated frames and encodes to JPEG for HTTP streaming | `frame_lock` (read-only) |
| **Flask Main Thread** | On request | Serves REST API, handles enrollment, returns JSON stats | Database connections |

> **Key Design Decision:** The camera thread and AI engine are fully decoupled. The camera captures and draws bounding boxes at 30fps using lightweight Haar cascades, while the heavier FaceNet model runs asynchronously at its own pace (~2-5fps depending on hardware). This ensures the video feed never stutters, even on low-spec machines.

---

## 🔬 Technical Deep Dive

### Face Detection Pipeline

SentriCam employs a **two-stage detection strategy** for optimal performance:

**Stage 1 — Fast Detection (30fps, Camera Thread):**
- Downscales frames to 480×360 for rapid processing
- Runs **dual Haar cascade classifiers** (frontal + profile) simultaneously
- Applies overlap deduplication using centroid distance
- Maps detections back to original 1280×720 coordinate space

**Stage 2 — Deep Recognition (2-5fps, Engine Thread):**
- Receives padded face crops from Stage 1
- Generates **128-dimensional FaceNet embeddings** using `DeepFace.represent()`
- Performs **brute-force cosine similarity search** against all enrolled embeddings
- Returns identity + confidence score per tracked face

### EMA Smoothing Algorithm

To eliminate the characteristic "jitter" of raw frame-by-frame detection, SentriCam applies **Exponential Moving Average smoothing** to bounding box coordinates:

```python
# Smoothing factor: 60% current detection, 40% previous position
x = int(0.6 * detected_x + 0.4 * smoothed_x)
y = int(0.6 * detected_y + 0.4 * smoothed_y)
w = int(0.6 * detected_w + 0.4 * smoothed_w)
h = int(0.6 * detected_h + 0.4 * smoothed_h)
```

Combined with a **15-frame TTL (Time-To-Live)** buffer, tracked faces persist through brief occlusions (e.g., turning away momentarily), preventing identity flickering.

### Cosine Distance Matching

```python
def findCosineDistance(a, b):
    """
    Cosine distance ∈ [0, 2]
    - 0.0 = identical embeddings (perfect match)
    - 0.4 = threshold (default cutoff for positive ID)
    - 2.0 = maximally dissimilar
    """
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **AI Model** | Google FaceNet (via DeepFace) | 128-dim facial embedding generation |
| **Computer Vision** | OpenCV 4.x | Camera I/O, Haar cascades, image processing |
| **Deep Learning** | TensorFlow 2.x | FaceNet model backend |
| **Web Framework** | Flask | REST API server + static file serving |
| **Database** | SQLite 3 | User storage, event logging, embedding persistence |
| **Serialization** | Python Pickle | Binary embedding storage in database |
| **Frontend** | Vanilla HTML/CSS/JS | Zero-dependency responsive dashboard |
| **Streaming** | MJPEG over HTTP | Browser-native video with no WebSocket overhead |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- A **webcam** (built-in or USB)
- ~2GB RAM free (for TensorFlow + FaceNet model)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SumukhBhat005/Sentricam-AI-Powered_Real-Time_FaceRecognition_Surveillance.git
cd Sentricam-AI-Powered_Real-Time_FaceRecognition_Surveillance

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install opencv-python numpy flask deepface tensorflow

# 4. Launch SentriCam
python app.py
```

### First Run

On the first launch, SentriCam will:
1. **Download the FaceNet model** (~90MB, one-time only)
2. **Initialize the SQLite database** (auto-created as `sentricam.db`)
3. **Detect and configure the webcam** (auto-tries multiple backends)

Once ready, open your browser and navigate to:

```
🌐 http://localhost:5000
```

### Quick Start Guide

1. **Enroll a person** — Enter a name and click "Capture" while facing the camera
2. **Add more photos** — Capture the same person from different angles for better accuracy
3. **Watch it work** — The live feed will identify enrolled people with green boxes and flag unknowns in red
4. **Check the logs** — All detection events are logged with timestamps and confidence scores

---

## 📡 API Reference

SentriCam exposes a RESTful API for integration with external systems:

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/api/stream` | MJPEG video stream (30fps) |
| `GET` | `/api/snapshot` | Single JPEG frame capture |
| `GET` | `/api/detections` | Current detection results (JSON) |
| `GET` | `/api/users` | List all enrolled users |
| `POST` | `/api/users/enroll` | Enroll a new face `{name: "..."}` |
| `DELETE` | `/api/users/<name>` | Remove a user and all their embeddings |
| `GET` | `/api/logs` | Recent detection event log |
| `GET` | `/api/stats` | Dashboard statistics (detections, alerts, uptime) |
| `GET` | `/api/settings` | Current configuration |
| `POST` | `/api/settings` | Update threshold, cooldown, quality |

### Example: Enroll a User via cURL

```bash
curl -X POST http://localhost:5000/api/users/enroll \
  -H "Content-Type: application/json" \
  -d '{"name": "Sumukh"}'
```

### Example: Get Live Detections

```bash
curl http://localhost:5000/api/detections
# Response:
# {
#   "detections": [
#     {"name": "Sumukh", "distance": 0.21},
#     {"name": "Unknown", "distance": 0.67}
#   ],
#   "has_unknown": true,
#   "camera_ok": true
# }
```

---

## 📁 Project Structure

```
SentriCam/
│
├── app.py                  # Flask web server — API routes, MJPEG streaming,
│                           # camera thread, EMA tracking, and frame annotation
│
├── face_engine.py          # AI engine — FaceNet embedding generation,
│                           # cosine similarity matching, async processing loop
│
├── database.py             # Data layer — SQLite schema, user CRUD,
│                           # detection logging, stats aggregation
│
├── utils.py                # Utilities — Timestamped logger
│
├── sentricam.db            # SQLite database (auto-generated)
│
├── website/                # Frontend dashboard
│   ├── index.html          # Dashboard layout — live feed, stats, user mgmt
│   ├── style.css           # Dark-glass UI theme with responsive design
│   └── app.js              # Client logic — polling, enrollment, settings
│
├── .gitignore
└── README.md               # This file
```

---

## ⚡ Performance Benchmarks

Measured on a mid-range laptop (Intel i5-12th Gen, 16GB RAM, no dedicated GPU):

| Metric | Value |
|:---|:---:|
| **Camera Capture** | 30 fps |
| **Face Detection (Haar)** | < 8ms per frame |
| **FaceNet Inference** | ~180ms per face |
| **Cosine Matching** (10 enrolled) | < 0.1ms |
| **End-to-End Latency** (detection → ID) | ~200ms |
| **MJPEG Stream Bandwidth** | ~2.5 MB/s @ 720p |
| **Memory Footprint** | ~800MB (with TensorFlow loaded) |
| **Database Write (event log)** | < 1ms |

> The decoupled architecture ensures the video feed **never drops below 30fps** regardless of how many faces are being processed by the AI engine.

---

## 🔒 Privacy & Security

SentriCam was built with a **privacy-first philosophy**:

- 🏠 **Fully Local** — All processing happens on your machine. No data is ever transmitted externally.
- 🚫 **No Cloud APIs** — No AWS, Azure, Google Cloud, or any third-party service dependencies.
- 🔐 **No Raw Images Stored** — Only mathematical embeddings (128-dimensional float vectors) are persisted — faces cannot be reconstructed from embeddings.
- 📴 **Offline Capable** — Works without an internet connection after initial model download.
- 🗄️ **Data Sovereignty** — The SQLite database file stays on your local filesystem. Delete it, and all biometric data is permanently erased.

> ⚠️ **Important:** SentriCam is designed for trusted local networks. Do not expose port 5000 to the public internet without implementing proper authentication and TLS encryption.

---

## 🗺️ Future Roadmap

- [ ] **GPU Acceleration** — CUDA/cuDNN support for 10x faster FaceNet inference
- [ ] **Multi-Camera Support** — Monitor multiple camera feeds simultaneously
- [ ] **Push Notifications** — Telegram/Discord/Email alerts for unknown face detections
- [ ] **Face Anti-Spoofing** — Liveness detection to prevent photo/video replay attacks
- [ ] **Cloud Deployment** — Docker containerization + PostgreSQL for production scaling
- [ ] **Mobile Companion App** — React Native app for remote monitoring
- [ ] **RTSP Support** — Connect IP cameras and NVR systems

---

## 📄 License

This project is open-source and available for educational and portfolio purposes. Feel free to fork and modify!

---

<div align="center">

### Built with ❤️ by [Sumukh Bhat](https://github.com/SumukhBhat005)

*A project demonstrating real-time AI, computer vision, multi-threaded systems design, and full-stack web development.*

⭐ **Star this repo if you found it interesting!**

</div>
