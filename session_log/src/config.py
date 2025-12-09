import os


class Config:
    # ASR service
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_URL = f"http://127.0.0.1:{API_PORT}/transcribe"

    # websocket
    WS_URL = f"ws://127.0.0.1:{API_PORT}/ws/realtime"

    # NATS
    NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
    LOG_SUBJECT = "asr.inference.stream"

    # MinIO (S3)
    S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://10.0.0.56:39001")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "admin")
    S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "password123")
    S3_BUCKET = "audio"

    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    MONGO_DB = "asr_data"
