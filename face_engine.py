import cv2
import numpy as np
import threading
import time
from deepface import DeepFace
from database import load_known_users
from utils import Logger

def findCosineDistance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class FaceEngine:
    def __init__(self, threshold=0.40):
        # We use threshold=0.40 for Facenet with Cosine Distance
        self.threshold = threshold
        self.known_faces = load_known_users() # {name: [emb1, emb2]}
        self.model_name = "Facenet"
        self.detector_backend = "opencv"
        
        # Thread sync
        self.lock = threading.Lock()
        self.latest_crops = []
        self._results = {}
        
        # Thread flag
        self.is_running = False

    def start(self):
        Logger.info("Starting Face Engine Thread...")
        self.is_running = True
        t = threading.Thread(target=self._process_loop, daemon=True)
        t.start()
        
        # Dummy call to initialize the model into memory preventing freeze on first frame
        Logger.info("Initializing Facenet (may take a moment on first run)...")
        try:
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.represent(img_path=dummy_img, model_name=self.model_name, detector_backend="skip", enforce_detection=False)
            Logger.info("Facenet loaded.")
        except Exception as e:
            pass

    def stop(self):
        self.is_running = False

    def update_crops(self, crops):
        """Called constantly by the UI thread to pass padded face crops {track_id, crop}."""
        with self.lock:
            self.latest_crops = crops

    def get_results(self):
        """Returns the dictionary of {track_id: {name, distance}}."""
        with self.lock:
            return dict(self._results)

    def force_reload_users(self):
        with self.lock:
            self.known_faces = load_known_users()

    def generate_embedding(self, frame):
        """Synchronous function for user management to enroll a face."""
        try:
            # We enforce detection here because we want to make sure an actual face is saved.
            objs = DeepFace.represent(img_path=frame, 
                                      model_name=self.model_name, 
                                      detector_backend=self.detector_backend,
                                      enforce_detection=True)
            if len(objs) > 0:
                # Return the embedding of the first face found
                return objs[0]["embedding"]
        except Exception as e:
            Logger.error(f"Error generating embedding: {e}")
        return None

    def _process_loop(self):
        while self.is_running:
            crops_to_process = []
            with self.lock:
                crops_to_process = self.latest_crops
                self.latest_crops = []
                
            if not crops_to_process:
                time.sleep(0.01)
                continue

            results_dict = {}
            for item in crops_to_process:
                track_id = item["track_id"]
                crop = item["crop"]
                
                try:
                    # Enforce detection = False because we ALREADY cropped it securely
                    objs = DeepFace.represent(img_path=crop, 
                                              model_name=self.model_name, 
                                              detector_backend="skip",
                                              enforce_detection=False)
                    
                    if len(objs) > 0:
                        embedding = objs[0]["embedding"]
                        best_match = "Unknown"
                        best_dist = float("inf")
                        
                        for name, embeddings_list in self.known_faces.items():
                            for stored_emb in embeddings_list:
                                dist = findCosineDistance(stored_emb, embedding)
                                if dist < best_dist:
                                    best_dist = dist
                                    if best_dist < self.threshold:
                                        best_match = name

                        results_dict[track_id] = {
                            "name": best_match,
                            "distance": round(best_dist, 2)
                        }
                except ValueError:
                    pass
                except Exception as e:
                    pass

            with self.lock:
                self._results = results_dict
