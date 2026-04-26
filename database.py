import sqlite3
import pickle
import time
from utils import Logger

DB_PATH = 'sentricam.db'

def get_connection():
    # check_same_thread=False is perfectly safe here considering our app structure
    # timeout=10 prevents 'database is locked' errors by gracefully waiting
    return sqlite3.connect(DB_PATH, timeout=10.0, check_same_thread=False)

def init_db():
    """Initialize the SQLite database with the users table."""
    Logger.info("Initializing database...")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            confidence REAL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(name, embedding):
    """Adds a new user and their embedding to the database.
    embedding: list or numpy array
    """
    serialized_embedding = pickle.dumps(embedding)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name, embedding) VALUES (?, ?)', (name, serialized_embedding))
    conn.commit()
    conn.close()
    Logger.info(f"User '{name}' added successfully to the database.")

def load_known_users():
    """Returns a lookup mapping: {name: [embedding1, embedding2], ...}"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT name, embedding FROM users')
    rows = cursor.fetchall()
    conn.close()

    known_faces = {}
    for name, serialized_embedding in rows:
        embedding = pickle.loads(serialized_embedding)
        if name not in known_faces:
            known_faces[name] = []
        known_faces[name].append(embedding)
    
    Logger.info(f"Loaded {len(known_faces.keys())} unique users from database.")
    return known_faces

def get_all_users():
    """Returns a list of {id, name} dicts (no embeddings) for the web UI."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, name FROM users')
    rows = cursor.fetchall()
    conn.close()
    # Group by name to show unique users with their IDs
    users = {}
    for uid, name in rows:
        if name not in users:
            users[name] = {"name": name, "ids": [], "photo_count": 0}
        users[name]["ids"].append(uid)
        users[name]["photo_count"] += 1
    return list(users.values())

def delete_user(name):
    """Removes all embeddings for a user by name."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE name = ?', (name,))
    conn.commit()
    conn.close()
    Logger.info(f"User '{name}' deleted from database.")

def add_detection_log(name, confidence):
    """Logs a face detection event."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO detection_logs (name, confidence, timestamp) VALUES (?, ?, ?)',
                   (name, confidence, timestamp))
    conn.commit()
    conn.close()

def get_recent_logs(limit=50):
    """Returns the most recent detection logs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, confidence, timestamp FROM detection_logs ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "confidence": round(r[2], 3) if r[2] else 0, "timestamp": r[3]} for r in rows]

def get_detection_stats():
    """Returns stats for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    today = time.strftime("%Y-%m-%d")
    cursor.execute('SELECT COUNT(*) FROM detection_logs WHERE timestamp LIKE ?', (today + '%',))
    today_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM detection_logs WHERE name = ? AND timestamp LIKE ?', ('Unknown', today + '%',))
    unknown_today = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(DISTINCT name) FROM users')
    total_users = cursor.fetchone()[0]
    conn.close()
    return {"detections_today": today_count, "unknown_alerts_today": unknown_today, "total_users": total_users}

