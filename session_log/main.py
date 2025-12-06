# main.py
import json
import asyncio
from fastapi import FastAPI, UploadFile
import nats
from nats.errors import TimeoutError

app = FastAPI()

# 全局 NATS 变量
nc = None
js = None

@app.on_event("startup")
async def startup_event():
    global nc, js
    # 连接到 K3s 中的 NATS 服务
    nc = await nats.connect("nats://nats:4222")
    js = nc.jetstream()
    print("Connected to NATS JetStream")

@app.on_event("shutdown")
async def shutdown_event():
    if nc:
        await nc.close()

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    # ... (省略 ASR 推理逻辑，获取 audio_bytes 和 transcript) ...
    request_id = "uuid-123"
    transcript = "测试文本"
    
    # 构建日志数据
    log_payload = {
        "request_id": request_id,
        "transcript": transcript,
        "meta": {"timestamp": 1234567890}
        # 注意：实际场景中，音频太大的话，建议先上传MinIO获取Key，
        # 再把Key放进MQ。这里假设直接传Key。
        "s3_key": f"raw/{request_id}.wav"
    }

    try:
        # 异步发送到 NATS Subject 'asr.logs.new'
        ack = await js.publish(
            "asr.logs.new", 
            json.dumps(log_payload).encode(), 
            timeout=5
        )
        print(f"Log published: {ack.stream}")
    except Exception as e:
        print(f"Failed to publish log: {e}")

    return {"id": request_id, "text": transcript}