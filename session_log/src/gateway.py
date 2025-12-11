import asyncio
import json
import base64
import time
import uuid
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import nats
from config import Config


# --- 会话管理器 ---
class ConnectionManager:
    def __init__(self):
        # 存储格式: { "session_id": {"ws": WebSocket, "history": "..."} }
        self.active_sessions = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_sessions[session_id] = {
            "ws": websocket,
            "history": "",  # 在网关层维护上下文，让 Worker 无状态
        }

    def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    async def send_text(self, session_id: str, text: str, latency: float):
        if session_id in self.active_sessions:
            ws = self.active_sessions[session_id]["ws"]
            try:
                await ws.send_json({"type": "update", "text": text, "latency": latency})
            except Exception:
                pass  # 连接可能已断开

    def update_history(self, session_id: str, new_text: str):
        if session_id in self.active_sessions:
            # 简单策略：追加文本，只保留最后 200 字符作为 Prompt
            current = self.active_sessions[session_id]["history"]
            updated = current + new_text
            self.active_sessions[session_id]["history"] = updated[-200:]

    def get_history(self, session_id: str):
        return self.active_sessions.get(session_id, {}).get("history", "")


manager = ConnectionManager()
server_state = {"nc": None, "js": None}


# --- NATS 消息处理 (收结果) ---
async def handle_asr_result(msg):
    """
    处理 ASR Worker 发回来的结果 (监听 asr.output)
    """
    try:
        data = json.loads(msg.data.decode())
        session_id = data.get("session_id")
        text = data.get("text")
        latency = data.get("latency", 0)

        if session_id and text:
            # 1. 更新网关维护的上下文
            manager.update_history(session_id, text)

            # 2. 推送给前端 Gradio
            await manager.send_text(session_id, text, latency)

        await msg.ack()
    except Exception as e:
        print(f"❌ Gateway Error handling NATS msg: {e}")


# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 连接 NATS
    print(f"🔌 [Gateway] Connecting to NATS: {Config.NATS_URL}")
    server_state["nc"] = await nats.connect(Config.NATS_URL)
    server_state["js"] = server_state["nc"].jetstream()

    # 2. 订阅 ASR 结果
    # 注意：Gateway 是广播接收，需要根据 session_id 自己做路由
    await server_state["js"].subscribe(
        "asr.output",
        cb=handle_asr_result,
        durable="gateway_router",  # 保证断连后能收到离线消息(可选)
    )
    print("✅ [Gateway] Listening for ASR results...")

    yield

    if server_state["nc"]:
        await server_state["nc"].close()


app = FastAPI(lifespan=lifespan)


# --- WebSocket 路由 (收音频) ---
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    # 为每个连接生成唯一的 Session ID
    session_id = str(uuid.uuid4())
    print(f"🔌 Client connected: {session_id}")

    await manager.connect(session_id, websocket)

    # 音频缓冲配置 (2秒切片)
    audio_buffer = bytearray()
    SAMPLE_RATE = 16000
    BYTES_PER_SEC = SAMPLE_RATE * 2  # int16 = 2 bytes
    THRESHOLD_BYTES = int(BYTES_PER_SEC * 2.0)

    try:
        while True:
            # 接收 Gradio 发来的原始字节流 (Int16 PCM)
            data = await websocket.receive_bytes()

            # 1. 缓冲
            audio_buffer.extend(data)

            # 2. 切片 & 发布
            if len(audio_buffer) >= THRESHOLD_BYTES:
                # 准备发送给 NATS 的 Payload
                # 获取当前会话的上下文
                prompt_text = manager.get_history(session_id)

                req_id = str(uuid.uuid4())
                payload = {
                    "req_id": req_id,
                    "session_id": session_id,
                    "audio_b64": base64.b64encode(audio_buffer).decode("utf-8"),
                    "previous_text": prompt_text,
                    "timestamp": time.time(),
                }

                # 发布到 asr.input，等待 Worker 抢单处理
                if server_state["js"]:
                    await server_state["js"].publish(
                        "asr.input", json.dumps(payload).encode()
                    )

                # 清空缓冲
                audio_buffer.clear()

            # 处理 EOF (可选)
            # if data == b"EOF": ...

    except WebSocketDisconnect:
        print(f"👋 Client disconnected: {session_id}")
        manager.disconnect(session_id)
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
        manager.disconnect(session_id)


if __name__ == "__main__":
    import uvicorn

    # 启动 Gateway，监听 8000 端口
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
