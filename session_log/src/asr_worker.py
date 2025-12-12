import asyncio
import json
import base64
import time
import os
import numpy as np
import nats
from nats.errors import TimeoutError
from faster_whisper import WhisperModel
from logger import setup_logger
from src.config import Config

DEVICE = os.getenv("ASR_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float16")

logger = setup_logger("asr-worker")


class ASRWorker:
    def __init__(self):
        self.nc = None
        self.js = None
        self.model = None

    def load_model(self):
        print(f"⏳ [ASR Worker] Loading Whisper Model ({DEVICE}/{COMPUTE_TYPE})...")
        try:
            self.model = WhisperModel("tiny", device=DEVICE, compute_type=COMPUTE_TYPE)
            print("✅ [ASR Worker] Model Loaded successfully!")
        except Exception as e:
            print(f"❌ [ASR Worker] CRITICAL: Model load failed - {e}")
            exit(1)

    def run_inference(self, audio_np, previous_text=""):
        if not self.model:
            return ""

        segments, info = self.model.transcribe(
            audio_np,
            beam_size=1,
            language="en",
            initial_prompt=previous_text,
            condition_on_previous_text=True,
            vad_filter=True,
        )
        result_text = "".join([s.text for s in segments])
        return result_text

    async def process_msg(self, msg):
        """
        process single NATS message
        message: {
            "req_id": "uuid...",
            "session_id": "user-session-123",
            "audio_b64": "base64_encoded_string...",
            "previous_text": "上一次识别的结果...",
            "timestamp": 1234567890
        }
        """
        try:
            payload = json.loads(msg.data.decode())
            req_id = payload.get("req_id", "unknown")
            session_id = payload.get("session_id", "unknown")

            logger.info(f"Start inference", extra={"req_id": req_id})
            start_time = time.time()

            # 1. 解码音频 (Base64 -> Float32 Numpy)
            audio_bytes = base64.b64decode(payload["audio_b64"])
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # 2. 获取上下文 (Prompt)
            # 在分布式架构中，上下文最好由网关维护并传过来，Worker 保持无状态
            previous_text = payload.get("previous_text", "")

            # 3. 执行推理 (跑在线程池中，避免阻塞 asyncio 循环)
            loop = asyncio.get_running_loop()
            new_text = await loop.run_in_executor(
                None, self.run_inference, audio_float32, previous_text
            )

            latency = round(time.time() - start_time, 3)
            logger.info(f"Inference done", extra={"req_id": req_id, "latency": latency})

            # 4. 如果有识别结果，发布到输出队列
            if new_text.strip():
                print(f"✅ Result [{session_id}]: '{new_text}' ({latency}s)")

                output_payload = {
                    "req_id": req_id,
                    "session_id": session_id,
                    "text": new_text,
                    "latency": latency,
                    "timestamp": time.time(),
                    # 可以在这里把 audio_b64 再次传下去，给存储服务存 MinIO
                    # 或者存储服务直接订阅 asr.input 也可以
                    "audio_b64": payload["audio_b64"],
                }

                # 发布到 'asr.output'，供：
                # 1. 存储服务 (Storage Worker) 存数据库
                # 2. 网关服务 (Gateway) 发回给前端
                await self.js.publish("asr.output", json.dumps(output_payload).encode())

            # 5. 确认消息 (Ack)
            await msg.ack()

        except Exception as e:
            print(f"❌ Error processing message: {e}")
            # 如果是数据格式错误，建议 Ack 掉，否则 NATS 会一直重发导致死循环
            # 如果是临时故障，可以 msg.nak()
            await msg.ack()

    async def start(self):
        # 1. 加载模型
        self.load_model()

        # 2. 连接 NATS
        print(f"🔌 Connecting to NATS: {Config.NATS_URL}")
        self.nc = await nats.connect(Config.NATS_URL)
        self.js = self.nc.jetstream()

        # 3. 创建 Stream (如果不存在)
        # 这里监听 asr.input.*
        try:
            await self.js.add_stream(name="ASR_INPUT", subjects=["asr.input"])
        except Exception:
            pass  # Stream 可能已存在

        # 4. 启动订阅 (Queue Group)
        # 关键点：使用 queue="asr_workers"
        # 这样如果你启动了 3 个 ASR Worker 副本，NATS 会自动负载均衡，
        # 每条音频只会被一个 Worker 处理，不会重复。
        sub = await self.js.subscribe(
            "asr.input", queue="asr_workers", cb=self.process_msg, manual_ack=True
        )

        print("🚀 ASR Worker started! Waiting for audio chunks...")

        # 保持运行
        try:
            await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass
        finally:
            await self.nc.close()


if __name__ == "__main__":
    worker = ASRWorker()
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        print("🛑 Worker stopped.")
