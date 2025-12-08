import os


class Config:
    # NATS
    NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
    LOG_SUBJECT = "asr.logs.new"

    # MinIO (S3)
    S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "admin")
    S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "password123")
    S3_BUCKET = "asr-audio-raw"

    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    MONGO_DB = "asr_data"
