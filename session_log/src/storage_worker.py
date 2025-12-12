import asyncio
import json
import base64
import time  # 之前代码缺少这个，会导致 strftime 报错
import os
import nats
import boto3
from pymongo import MongoClient
from io import BytesIO
from src.config import Config

# --- 初始化资源 (全局) ---

# 1. S3 / MinIO 客户端
try:
    s3 = boto3.client(
        "s3",
        endpoint_url=Config.S3_ENDPOINT,
        aws_access_key_id=Config.S3_ACCESS_KEY,
        aws_secret_access_key=Config.S3_SECRET_KEY,
    )
    # 简单的连通性测试
    s3.list_buckets()
    print("✅ [Storage] MinIO connected.")
except Exception as e:
    print(f"❌ [Storage] MinIO connection failed: {e}")
    # 注意：生产环境这里应该退出，否则后面会一直报错

# 2. MongoDB 客户端
try:
    mongo_client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
    # 触发一次连接检查
    mongo_client.server_info()
    db = mongo_client[Config.MONGO_DB]
    collection = db["transcriptions"]  # 改个更贴切的表名
    print("✅ [Storage] MongoDB connected.")
except Exception as e:
    print(f"❌ [Storage] MongoDB connection failed: {e}")


async def process_msg(msg):
    """
    处理来自 asr.output 的消息
    """
    try:
        data = json.loads(msg.data.decode())

        # 字段对齐：ASR Worker 发送的是 "req_id"
        req_id = data.get("req_id", "unknown_id")
        session_id = data.get("session_id", "default_session")

        # print(f"📥 Archiving {req_id}...")

        # ---------------------------------------------------------
        # 1. 处理音频 (存入 MinIO)
        # ---------------------------------------------------------
        s3_key = ""
        if "audio_b64" in data and data["audio_b64"]:
            try:
                # Base64 -> Bytes
                audio_bytes = base64.b64decode(data["audio_b64"])

                # 生成存储路径: yyyy/mm/dd/session_id/req_id.wav
                date_prefix = time.strftime("%Y/%m/%d")
                s3_key = f"{date_prefix}/{session_id}/{req_id}.wav"

                # 上传 (使用 upload_fileobj 内存上传，不落盘)
                s3.upload_fileobj(
                    BytesIO(audio_bytes),
                    Config.S3_BUCKET,
                    s3_key,
                    ExtraArgs={"ContentType": "audio/wav"},
                )
            except Exception as s3_e:
                print(f"⚠️ S3 Upload Failed for {req_id}: {s3_e}")
                # 即使 S3 失败，我们也可能想保留 MongoDB 记录，或者选择 nak 重试
                # 这里选择记录错误但继续执行

        # ---------------------------------------------------------
        # 2. 处理元数据 (存入 MongoDB)
        # ---------------------------------------------------------
        # 移除 heavy 的 audio 数据，只存路径
        if "audio_b64" in data:
            del data["audio_b64"]

        data["s3_key"] = s3_key
        data["s3_bucket"] = Config.S3_BUCKET
        data["archived_at"] = time.time()

        # 写入 Mongo
        collection.insert_one(data)

        # ---------------------------------------------------------
        # 3. 确认消息 (Ack)
        # ---------------------------------------------------------
        await msg.ack()
        # print(f"✅ Saved {req_id}")

    except Exception as e:
        print(
            f"❌ Critical Error processing {req_id if 'req_id' in locals() else 'msg'}: {e}"
        )
        # 如果是严重的逻辑错误或数据库断连，告诉 NATS 稍后重试
        # 注意：如果数据本身格式是错的，NAK 会导致死循环，需要仔细权衡
        await msg.nak()


async def main():
    print(f"🔌 [Storage] Connecting to NATS: {Config.NATS_URL}")
    nc = await nats.connect(Config.NATS_URL)
    js = nc.jetstream()

    # 确保存储相关的 Stream 存在
    # 我们可以复用 ASR_WORKER 创建的，或者建立一个专门用于持久化的 Stream
    # 这里假设我们监听 asr.output
    try:
        await js.add_stream(name="ASR_ARCHIVE", subjects=["asr.output"])
    except Exception:
        pass  # Stream 可能已存在

    # --- 关键修改：使用 Queue Group (队列组) ---
    # queue="storage_workers" 意味着：
    # 即使你启动了 10 个 Storage 服务副本，每条消息也只会被其中 1 个收到。
    # 如果不加这个参数，每条消息会被所有副本收到，导致数据库存了 10 份重复数据。

    print("🚀 Storage Worker started, listening to 'asr.output'...")

    await js.subscribe(
        "asr.output", queue="storage_workers", cb=process_msg, manual_ack=True
    )

    # 保持运行
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        await nc.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Storage Worker stopped.")
