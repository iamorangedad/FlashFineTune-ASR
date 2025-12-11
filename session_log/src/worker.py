import asyncio
import json
import base64
import nats
import boto3
from pymongo import MongoClient
from io import BytesIO
from .config import Config

# 初始化 S3 客户端
s3 = boto3.client(
    "s3",
    endpoint_url=Config.S3_ENDPOINT,
    aws_access_key_id=Config.S3_ACCESS_KEY,
    aws_secret_access_key=Config.S3_SECRET_KEY,
)

# 初始化 Mongo 客户端
mongo_client = MongoClient(Config.MONGO_URI)
db = mongo_client[Config.MONGO_DB]
collection = db["logs"]


async def process_msg(msg):
    data = json.loads(msg.data.decode())
    req_id = data["request_id"]

    try:
        print(f"📥 Processing {req_id}...")

        # 1. 处理音频：Base64 -> Bytes
        audio_bytes = base64.b64decode(data["audio_b64"])

        # 2. 上传到 MinIO
        s3_key = f"{time.strftime('%Y/%m/%d')}/{req_id}.wav"
        s3.upload_fileobj(BytesIO(audio_bytes), Config.S3_BUCKET, s3_key)

        # 3. 准备元数据 (移除 heavy 的 audio 数据)
        del data["audio_b64"]
        data["s3_key"] = s3_key
        data["s3_bucket"] = Config.S3_BUCKET

        # 4. 写入 MongoDB
        collection.insert_one(data)

        # 5. 确认消息 (Ack)
        await msg.ack()
        print(f"✅ Saved {req_id}")

    except Exception as e:
        print(f"❌ Error processing {req_id}: {e}")
        # 这里可以选择 msg.nak() 让 NATS稍后重试


async def main():
    nc = await nats.connect(Config.NATS_URL)
    js = nc.jetstream()
    stream_name = "ASR_INFERENCE"
    try:
        await js.add_stream(name=stream_name, subjects=[Config.LOG_SUBJECT])
    except Exception as e:
        print(f"⚠️ Warning during stream creation (might already exist): {e}")
    # 持久化订阅
    durable_name = "asr_log_processor"
    psub = await js.pull_subscribe(Config.LOG_SUBJECT, durable=durable_name)
    print("🚀 Worker started, waiting for logs...")
    while True:
        try:
            msgs = await psub.fetch(1, timeout=5)
            for msg in msgs:
                await process_msg(msg)
        except nats.errors.TimeoutError:
            continue
        except Exception as e:
            print(f"❌ NATS Error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
