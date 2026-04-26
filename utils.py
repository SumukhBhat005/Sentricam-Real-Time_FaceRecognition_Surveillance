import time

class Logger:
    @staticmethod
    def info(msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [INFO] {msg}")

    @staticmethod
    def warning(msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [WARN] {msg}")

    @staticmethod
    def error(msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [ERROR] {msg}")
