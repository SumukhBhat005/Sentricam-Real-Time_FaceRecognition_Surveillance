# SentriCam 👁️

SentriCam is an AI-powered biometric surveillance system designed to provide real-time face recognition and monitoring. It runs locally, keeping your data secure, and uses state-of-the-art models for accurate and reliable face tracking and identification.

## 🚀 Features

- **Real-Time Detection:** Live 30fps MJPEG streaming with hardware acceleration.
- **Accurate Recognition:** Uses Google's FaceNet and the DeepFace framework.
- **Multi-Angle Enrollment:** Register users with 5 different head angles for high precision.
- **Instant Alerts:** Identifies unknown individuals and alerts the dashboard instantly.
- **Activity Logging:** Searchable database logs of all detection events.
- **Jitter-Free Tracking:** Smooth tracking using dual-cascade detection paired with an EMA filter.
- **Local Data Processing:** No cloud APIs. Your data remains strictly on your device.

## 🛠️ Technology Stack

- **Backend:** Flask (Python)
- **Database:** SQLite
- **AI/ML:** OpenCV, DeepFace, FaceNet, TensorFlow
- **Frontend:** HTML, CSS, JavaScript (Vanilla, no bloated frameworks)

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SentriCam.git
   cd SentriCam
   ```

2. **Set up a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   Ensure you have Python 3.8+ installed. Then run:
   ```bash
   pip install opencv-python numpy flask deepface
   ```

4. **Run the Server:**
   ```bash
   python app.py
   ```

5. **Access the Dashboard:**
   Open your browser and navigate to `http://localhost:5000` to access the SentriCam Control Panel.

## ⚙️ Configuration

You can tweak system settings inside the application dashboard:
- **Detection Threshold:** Adjust sensitivity of the facial matching algorithm.
- **Stream Quality:** Balance performance versus visual fidelity.

## 🛡️ Privacy & Security

SentriCam encodes and matches your biometric data purely locally using an offline DeepFace model. For maximum security, we recommend not exposing the application publicly on the internet but keeping it contained in a trusted local network.

## 📄 License
This project is for educational and portfolio purposes. Feel free to fork and modify!
