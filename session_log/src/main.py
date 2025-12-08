import uuid
import json
import time
import base64
from fastapi import FastAPI, UploadFile, BackgroundTasks
from contextlib import asynccontextmanager
import nats
from .config import Config

# 全局 NATS 连接
nc = None
js = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时连接 NATS
    global nc, js
    try:
        nc = await nats.connect(Config.NATS_URL)
        js = nc.jetstream()
        print("✅ NATS Connected")
    except Exception as e:
        print(f"❌ NATS Connection failed: {e}")
    yield
    # 关闭时断开
    if nc:
        await nc.close()


app = FastAPI(lifespan=lifespan)


# 模拟 ASR 推理函数 (请替换为真实的 Whisper/Kaldi 调用)
def run_inference(audio_bytes):
    # import faster_whisper ...
    return "这是一个测试的语音转录结果", 0.98


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # 1. 读取音频 (注意内存，若文件过大需流式处理)
    audio_data = await file.read()

    # 2. 执行推理 (GPU)
    text, confidence = run_inference(audio_data)

    duration = time.time() - start_time

    # 3. 异步发送日志到 NATS
    # 将音频转为 Base64 以便通过 JSON 传输 (或者直接发二进制)
    # 注意：NATS 默认限制 1MB，需在 YAML 中调大 max_payload
    if js:
        payload = {
            "request_id": request_id,
            "timestamp": start_time,
            "duration": duration,
            "text": text,
            "confidence": confidence,
            "filename": file.filename,
            "audio_b64": base64.b64encode(audio_data).decode("utf-8"),
        }
        try:
            # 异步发布，不等待 Ack
            await js.publish(Config.LOG_SUBJECT, json.dumps(payload).encode())
        except Exception as e:
            print(f"Log publish failed: {e}")

    return {"request_id": request_id, "text": text, "latency": duration}
