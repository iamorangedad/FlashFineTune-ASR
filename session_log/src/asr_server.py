import uuid
import json
import time
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import nats

# 引入 faster-whisper
from faster_whisper import WhisperModel
from config import Config

# 全局状态
server_state = {"nc": None, "js": None, "model": None}  # 模型实例


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 连接 NATS
    try:
        server_state["nc"] = await nats.connect(Config.NATS_URL)
        server_state["js"] = server_state["nc"].jetstream()
        print("✅ [Server] NATS Connected")
    except Exception as e:
        print(f"❌ [Server] NATS Connection failed: {e}")

    # 2. 加载 Whisper 模型 (建议使用 'tiny', 'base', 'small' 以保证 CPU/低端 GPU 的实时性)
    # device="cuda" if you have GPU, else "cpu"
    # compute_type="float16" for GPU, "int8" for CPU
    print("⏳ [Server] Loading Faster-Whisper model...")
    try:
        server_state["model"] = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ [Server] Model Loaded")
    except Exception as e:
        print(f"❌ [Server] Model Load Failed: {e}")

    yield

    if server_state["nc"]:
        await server_state["nc"].close()


app = FastAPI(lifespan=lifespan)


def run_whisper_inference(audio_np, previous_text):
    """
    同步的推理函数，将在 executor 中运行
    :param audio_np: float32 的 numpy 数组
    :param previous_text: 上一段的文本，用作 prompt
    """
    model = server_state["model"]
    if not model:
        return ""

    # initial_prompt 是核心：它告诉模型上文说了什么
    segments, info = model.transcribe(
        audio_np,
        beam_size=1,  # 实时流一般设为1以追求速度
        language="zh",  # 强制中文，或去掉让它自动检测
        initial_prompt=previous_text,  # 注入上下文
        condition_on_previous_text=True,
        vad_filter=True,  # 开启 VAD 过滤静音
    )

    result_text = "".join([s.text for s in segments])
    return result_text


async def log_to_nats(payload):
    if server_state["js"]:
        try:
            await server_state["js"].publish(
                Config.LOG_SUBJECT, json.dumps(payload).encode()
            )
        except Exception:
            pass


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    await websocket.accept()
    request_id = str(uuid.uuid4())
    print(f"🔌 [WS] Connected: {request_id}")

    # 状态变量
    audio_buffer = bytearray()
    history_text = ""  # 保存所有识别出的历史文本，用于下一次的 prompt

    # 策略配置
    SAMPLE_RATE = 16000
    # 阈值：积累多少音频推理一次？
    # 16k采样 * 2字节(int16) * 2秒 = 64000 bytes
    CHUNK_DURATION_SEC = 2.0
    BYTES_PER_SEC = SAMPLE_RATE * 2
    THRESHOLD_BYTES = int(BYTES_PER_SEC * CHUNK_DURATION_SEC)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                audio_buffer.extend(message["bytes"])

                # 当缓冲区填满一定时长（如2秒）
                if len(audio_buffer) >= THRESHOLD_BYTES:
                    inference_start = time.time()

                    # --- 1. 数据转换 ---
                    # Client 发来的是 int16, Whisper 需要 float32 [-1, 1]
                    audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    # --- 2. 推理 ---
                    loop = asyncio.get_running_loop()
                    # 传入 history_text 作为 prompt
                    new_text = await loop.run_in_executor(
                        None, run_whisper_inference, audio_float32, history_text
                    )

                    # --- 3. 处理结果 ---
                    if new_text.strip():
                        # 更新上下文：这里有一个简单的策略
                        # 我们可以只保留最近的几句话作为 prompt，防止 prompt 无限长
                        history_text += new_text
                        # 限制 prompt 长度，保留最后200字符即可
                        prompt_context = history_text[-200:]

                        # 发送给前端
                        await websocket.send_json(
                            {
                                "type": "update",
                                "text": new_text,
                                "latency": round(time.time() - inference_start, 3),
                            }
                        )

                        # NATS 日志
                        asyncio.create_task(
                            log_to_nats(
                                {
                                    "req_id": request_id,
                                    "text_chunk": new_text,
                                    "timestamp": time.time(),
                                }
                            )
                        )

                    # --- 4. 清理 ---
                    # 简单策略：直接清空 buffer，准备接收下一个 2 秒
                    # 进阶策略：可以使用 Overlap (重叠窗口)，但这需要更复杂的去重逻辑
                    audio_buffer.clear()

            elif "text" in message and message["text"] == "EOF":
                # 处理剩余的一点点音频
                if len(audio_buffer) > 0:
                    audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    loop = asyncio.get_running_loop()
                    last_text = await loop.run_in_executor(
                        None, run_whisper_inference, audio_float32, history_text
                    )
                    if last_text.strip():
                        await websocket.send_json({"type": "update", "text": last_text})

                await websocket.send_json({"type": "finish"})
                break

    except WebSocketDisconnect:
        print(f"👋 [WS] Disconnected: {request_id}")
    except Exception as e:
        print(f"❌ [WS] Error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn

    # 注意：加载模型需要时间，可能需要几秒钟启动
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
