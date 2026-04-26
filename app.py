"""
SentriCam Web Server
Flask API that wraps the existing face engine, camera, and database.
Optimized for low-spec PCs with smooth 30fps MJPEG streaming.
"""
import cv2
import time
import threading
import numpy as np
from flask import Flask, send_from_directory, jsonify, request, Response
from database import (
    init_db, add_user, load_known_users, get_all_users,
    delete_user, add_detection_log, get_recent_logs, get_detection_stats,
    get_connection
)
from face_engine import FaceEngine
from utils import Logger

app = Flask(__name__, static_folder='website', static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# ── Globals ──────────────────────────────────────────────────────────
camera = None
camera_lock = threading.Lock()
engine = None
start_time = time.time()

def draw_reticle(img, x, y, w, h, color, thickness=2, length=20):
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

# Fast detector for instant UI drawing (no lag) — OpenCV Haar cascades
_haar_front = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
_haar_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# UI state for smoothing
smoothed_boxes = {}
tracked_boxes_ttl = {}

# Frame buffers (protected by frame_lock)
frame_lock = threading.Lock()
last_raw_frame = None           # Latest raw camera frame (30fps)
last_annotated_frame = None     # Latest frame with detection boxes drawn
last_results = []               # Latest detection results from engine
last_ui_detections = []         # Fast-tracked detections for the UI dashboard

# Settings (mutable at runtime)
settings = {
    "threshold": 0.40,
    "alert_cooldown": 3.0,
    "camera_index": 0,
    "snapshot_quality": 82,       # JPEG quality for stream (higher = sharper)
    "target_fps": 30,             # Target FPS for camera + stream
}

# Detection dedup for logging
_logged_names = {}
_LOG_COOLDOWN = 5
_camera_ok = False


def init_camera(index=0):
    """Initialize the camera. Tries multiple backends and indices."""
    global camera, _camera_ok
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass

        attempts = []
        for idx in [index, 0, 1, 2]:
            attempts.append((idx, cv2.CAP_DSHOW))
            attempts.append((idx, None))

        tried = set()
        for idx, backend in attempts:
            key = (idx, backend)
            if key in tried:
                continue
            tried.add(key)

            try:
                cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Set resolution and try to set FPS
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        # Reduce buffer size so we always get the freshest frame
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        camera = cap
                        _camera_ok = True
                        bname = "DSHOW" if backend is not None else "DEFAULT"
                        Logger.info(f"[WEB] Camera opened at index {idx} ({bname})")
                        return True
                cap.release()
            except Exception as e:
                Logger.warning(f"[WEB] Camera idx={idx} failed: {e}")

        Logger.error("[WEB] No camera found!")
        _camera_ok = False
        camera = None
        return False


# ── Camera Thread (30fps) ────────────────────────────────────────────
# Reads camera frames as fast as possible and annotates with latest results.
# This is DECOUPLED from the face engine — engine runs at its own pace.

def camera_read_loop():
    """Reads camera at ~30fps, annotates with latest detections, stores frames."""
    global last_raw_frame, last_annotated_frame, _camera_ok

    retry_counter = 0
    target_delay = 1.0 / settings["target_fps"]  # ~33ms for 30fps

    identified_faces = {}      # {track_id: name}
    identified_faces_ttl = {}  # {track_id: frames_left}


    while True:
        if camera is None or not _camera_ok:
            retry_counter += 1
            if retry_counter % 10 == 0:
                Logger.info("[WEB] Retrying camera init...")
                init_camera(settings["camera_index"])
            time.sleep(1)
            continue

        loop_start = time.time()

        # Read frame
        with camera_lock:
            try:
                ret, frame = camera.read()
            except Exception:
                ret = False

        if not ret or frame is None:
            _camera_ok = False
            time.sleep(0.5)
            continue

        retry_counter = 0
        frame = cv2.flip(frame, 1)

        # Store raw frame for enrollment
        with frame_lock:
            last_raw_frame = frame

        annotated = frame.copy()
        now = time.time()
        
        # 1) Instant Face Detection (Runs at 30fps for 0-delay UI tracking)
        # Using OpenCV Haar cascades — fast, no extra dependencies
        fh_orig, fw_orig = frame.shape[:2]
        detect_w, detect_h = 480, 360
        scale_x = fw_orig / detect_w
        scale_y = fh_orig / detect_h
        small_gray = cv2.cvtColor(cv2.resize(frame, (detect_w, detect_h)), cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(small_gray, small_gray)

        fronts = _haar_front.detectMultiScale(small_gray, scaleFactor=1.08, minNeighbors=3, minSize=(25, 25))
        profiles = _haar_profile.detectMultiScale(small_gray, scaleFactor=1.08, minNeighbors=3, minSize=(25, 25))

        all_fast_faces = []
        for (x, y, w, h) in (list(fronts) + list(profiles)):
            all_fast_faces.append((x, y, w, h))

        # Simple overlap dedup
        fast_faces = []
        for b in all_fast_faces:
            bx, by, bw, bh = b
            duplicate = False
            for fx, fy, fw, fh in fast_faces:
                d = ((bx+bw/2 - fx-fw/2)**2 + (by+bh/2 - fy-fh/2)**2)**0.5
                if d < max(bw, fw) * 0.5:
                    duplicate = True
                    break
            if not duplicate:
                fast_faces.append(b)
        
        drawn_fast_boxes = []
        new_smoothed_boxes = {}
        new_ttl = {}
        used_fast_indices = set()

        # 1a) Match existing tracks to current fast_faces to apply EMA smoothing
        for track_id, (sx, sy, sw, sh) in smoothed_boxes.items():
            best_face_idx = -1
            best_dist = float('inf')
            for i, (fx, fy, fw, fh) in enumerate(fast_faces):
                if i in used_fast_indices: continue
                x, y, w, h = int(fx*scale_x), int(fy*scale_y), int(fw*scale_x), int(fh*scale_y)
                cx1, cy1 = sx + sw/2, sy + sh/2
                cx2, cy2 = x + w/2, y + h/2
                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                if dist < 250 and dist < best_dist:
                    best_dist = dist
                    best_face_idx = i
            
            if best_face_idx != -1:
                fx, fy, fw, fh = fast_faces[best_face_idx]
                x, y, w, h = int(fx*scale_x), int(fy*scale_y), int(fw*scale_x), int(fh*scale_y)
                x = int(0.6 * x + 0.4 * sx)
                y = int(0.6 * y + 0.4 * sy)
                w = int(0.6 * w + 0.4 * sw)
                h = int(0.6 * h + 0.4 * sh)
                new_smoothed_boxes[track_id] = (x, y, w, h)
                new_ttl[track_id] = 15 # Grant 15 frames (~0.5s) of memory
                used_fast_indices.add(best_face_idx)
            else:
                ttl = tracked_boxes_ttl.get(track_id, 0) - 1
                if ttl > 0:
                    new_smoothed_boxes[track_id] = (sx, sy, sw, sh)
                    new_ttl[track_id] = ttl
                
        # 1b) Register any new tracked faces
        next_id = max(smoothed_boxes.keys(), default=0) + 1 if smoothed_boxes else 1
        for i, (fx, fy, fw, fh) in enumerate(fast_faces):
            if i not in used_fast_indices:
                x, y, w, h = int(fx*scale_x), int(fy*scale_y), int(fw*scale_x), int(fh*scale_y)
                new_smoothed_boxes[next_id] = (x, y, w, h)
                new_ttl[next_id] = 15
                next_id += 1

        # 2) Extract crops and send to engine (non-blocking)
        engine_payload = []
        for track_id, (x, y, w, h) in new_smoothed_boxes.items():
            # Pad slightly for better recognition
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2].copy()
                engine_payload.append({"track_id": track_id, "crop": crop})
                
        if engine and engine_payload:
            engine.update_crops(engine_payload)

        # Get latest engine results
        results_dict = engine.get_results() if engine else {}

        # 3) Draw instant tracking boxes and assign identities
        text_queue = []
        current_ui_detections = []
        
        # Cleanup old identities
        for tid in list(identified_faces.keys()):
            if tid not in new_smoothed_boxes:
                del identified_faces[tid]
                if tid in identified_faces_ttl:
                    del identified_faces_ttl[tid]
        
        for track_id, (x, y, w, h) in new_smoothed_boxes.items():
            drawn_fast_boxes.append((x, y, w, h))
            
            match_name = "Scanning..."
            match_dist = None

            # Has the engine returned a result for this exact track ID?
            if track_id in results_dict:
                res = results_dict[track_id]
                match_name = res["name"]
                match_dist = res["distance"]
                
                # Check if we should log it
                if track_id not in identified_faces or identified_faces[track_id] == "Scanning...":
                    if match_name != "Unknown" and match_name != "Scanning...":
                        try:
                            # Verify cooldown
                            if match_name not in _logged_names or (now - _logged_names[match_name]) > _LOG_COOLDOWN:
                                _logged_names[match_name] = now
                                add_detection_log(match_name, match_dist)
                        except Exception:
                            pass
                
                identified_faces[track_id] = match_name
                identified_faces_ttl[track_id] = 15 # Grant visual memory
            else:
                # Use memory if engine hasn't returned new data yet
                if track_id in identified_faces and identified_faces_ttl.get(track_id, 0) > 0:
                    match_name = identified_faces[track_id]
                    identified_faces_ttl[track_id] -= 1
            
            color = (50, 220, 50)  # Lime green (known)
            if match_name == "Unknown": color = (40, 40, 220)  # Deep Crimson red
            elif match_name == "Scanning...": color = (0, 165, 255)  # Orange
                
            draw_reticle(annotated, x, y, w, h, color, thickness=2, length=20)
            
            ui_name = match_name
            if match_name not in ["Unknown", "Scanning..."]:
                ui_name = match_name.title()
                
            current_ui_detections.append({"name": ui_name, "distance": match_dist if match_dist else float("inf")})
            
            # Modern label formatting
            if match_name == "Unknown":
                label = "UNKNOWN"
            elif match_name == "Scanning...":
                label = "SCANNING"
            else:
                label = match_name.upper()
                if match_dist is not None and match_dist != float("inf"):
                    label += f" [{match_dist}]"
            
            # Draw solid dark label background directly (no alpha blend needed)
            (txt_w, txt_h), bk = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            pad = 6
            cv2.rectangle(annotated, (x, y - txt_h - pad*2), (x + txt_w + pad*2, y), color, -1)
            cv2.putText(annotated, label, (x + pad, y - pad), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            text_queue.append((x, y, label, color))
        
        smoothed_boxes.clear()
        smoothed_boxes.update(new_smoothed_boxes)
        tracked_boxes_ttl.clear()
        tracked_boxes_ttl.update(new_ttl)

        with frame_lock:
            last_annotated_frame = annotated
            last_results = list(results_dict.values())
            last_ui_detections = current_ui_detections

        # Sleep to maintain target FPS
        elapsed = time.time() - loop_start
        sleep_time = target_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ── MJPEG Stream ─────────────────────────────────────────────────────

def generate_mjpeg():
    """Generator that yields MJPEG frames at ~30fps."""
    target_delay = 1.0 / settings["target_fps"]

    while True:
        frame_start = time.time()

        with frame_lock:
            frame = last_annotated_frame

        if frame is None:
            # Blank frame while waiting for camera
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if not _camera_ok:
                cv2.putText(frame, "No Camera Found", (170, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 120), 2)
                cv2.putText(frame, "Retrying...", (230, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 90), 1)
            else:
                cv2.putText(frame, "Initializing...", (215, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        _, jpeg = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, settings["snapshot_quality"]])

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        elapsed = time.time() - frame_start
        sleep_time = target_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ── Static Files ─────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('website', 'index.html')


# ── API Endpoints ────────────────────────────────────────────────────

@app.route('/api/stream')
def stream():
    """MJPEG video stream — smooth 30fps, handled natively by the browser."""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache, no-store'})


@app.route('/api/snapshot')
def snapshot():
    """Single JPEG frame — fallback for browsers that don't support MJPEG."""
    with frame_lock:
        frame = last_annotated_frame

    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Feed", (180, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    _, jpeg = cv2.imencode('.jpg', frame,
                           [cv2.IMWRITE_JPEG_QUALITY, settings["snapshot_quality"]])
    return Response(jpeg.tobytes(), mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-cache, no-store'})


@app.route('/api/detections')
def detections():
    """Returns current face detection results synced to the fast UI tracker."""
    with frame_lock:
        results = list(last_ui_detections)
        
    has_unknown = any(r["name"] == "Unknown" for r in results)
    return jsonify({"detections": results, "has_unknown": has_unknown,
                    "camera_ok": _camera_ok})


@app.route('/api/users', methods=['GET'])
def list_users():
    return jsonify({"users": get_all_users()})


@app.route('/api/users/enroll', methods=['POST'])
def enroll_user():
    """Uses the latest raw frame for embedding (thread-safe)."""
    data = request.get_json()
    name = data.get("name", "").strip() if data else ""

    if not name:
        return jsonify({"error": "Name is required"}), 400

    with frame_lock:
        frame = last_raw_frame.copy() if last_raw_frame is not None else None

    if frame is None:
        return jsonify({"error": "Camera not available. Wait for the feed to start."}), 500

    embedding = engine.generate_embedding(frame)

    if embedding is not None:
        add_user(name, embedding)
        engine.force_reload_users()
        return jsonify({"success": True, "message": f"Photo captured for {name}"})
    else:
        return jsonify({"error": "No face detected. Make sure your face is clearly visible."}), 400


@app.route('/api/users/<name>', methods=['DELETE'])
def remove_user(name):
    delete_user(name)
    engine.force_reload_users()
    return jsonify({"success": True, "message": f"User '{name}' removed"})


@app.route('/api/logs')
def logs():
    limit = request.args.get('limit', 50, type=int)
    return jsonify({"logs": get_recent_logs(limit)})


@app.route('/api/stats')
def stats():
    s = get_detection_stats()
    
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT name, timestamp FROM detection_logs ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        if row:
            time_str = row[1].split(' ')[1][:5]
            s["last_seen"] = f"{row[0]} ({time_str})"
        else:
            s["last_seen"] = "-"
    except Exception:
        s["last_seen"] = "-"

    s["uptime_seconds"] = int(time.time() - start_time)
    s["camera_ok"] = _camera_ok
    return jsonify(s)


@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(settings)


@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    if "threshold" in data:
        settings["threshold"] = float(data["threshold"])
        engine.threshold = settings["threshold"]
    if "alert_cooldown" in data:
        settings["alert_cooldown"] = float(data["alert_cooldown"])
    if "snapshot_quality" in data:
        settings["snapshot_quality"] = max(20, min(95, int(data["snapshot_quality"])))

    return jsonify({"success": True, "settings": settings})


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    Logger.info("=" * 50)
    Logger.info("  SentriCam Web Server Starting...")
    Logger.info("=" * 50)

    init_db()

    engine = FaceEngine(threshold=settings["threshold"])
    engine.start()

    init_camera(settings["camera_index"])

    def safe_camera_loop():
        while True:
            try:
                camera_read_loop()
            except Exception as e:
                Logger.error(f"CRITICAL: Camera loop crashed! Recovering... Error: {e}")
                time.sleep(1)

    # Camera read thread — 30fps capture + annotation (FAST)
    threading.Thread(target=safe_camera_loop, daemon=True).start()

    Logger.info("Open http://localhost:5000 in your browser")
    Logger.info("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
