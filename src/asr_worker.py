import asyncio
import json
import base64
import time
import os
import wave
import numpy as np
import nats
from faster_whisper import WhisperModel
from src.logger import setup_logger
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
        logger.info(
            f"⏳ [ASR Worker] Loading Whisper Model ({DEVICE}/{COMPUTE_TYPE})..."
        )
        try:
            self.model = WhisperModel("tiny", device=DEVICE, compute_type=COMPUTE_TYPE)
            logger.info("✅ [ASR Worker] Model Loaded successfully!")
        except Exception as e:
            logger.critical(f"❌ [ASR Worker] CRITICAL: Model load failed - {e}")
            exit(1)

    def run_inference(self, audio_np, previous_text="", req_id="N/A"):
        if not self.model:
            return ""

        max_amp = np.max(np.abs(audio_np))
        avg_amp = np.mean(np.abs(audio_np))

        if max_amp < 0.005:
            logger.warning(
                f"🔇 Audio is too quiet (Max: {max_amp:.4f}). VAD will likely ignore it.",
                extra={"req_id": req_id, "vol_max": float(max_amp)},
            )
            return ""

        logger.info(
            f"🎤 Processing Audio: MaxVol={max_amp:.3f}, AvgVol={avg_amp:.3f}",
            extra={"req_id": req_id},
        )

        try:
            segments_gen, info = self.model.transcribe(
                audio_np,
                beam_size=1,
                language="en",
                initial_prompt=previous_text,
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            segments = list(segments_gen)
            if not segments:
                logger.warning(
                    f"⚠️ Whisper finished but found NO segments. (VAD likely filtered it out)",
                    extra={"req_id": req_id},
                )
                return ""

            result_text = "".join([s.text for s in segments])
            return result_text

        except Exception as e:
            logger.error(f"Inference Error: {e}", extra={"req_id": req_id})
            return ""

    async def process_msg(self, msg):
        try:
            payload = json.loads(msg.data.decode())
            req_id = payload.get("req_id", "unknown")
            session_id = payload.get("session_id", "unknown")
            start_time = time.time()
            # 1. Decode Audio
            audio_bytes = base64.b64decode(payload["audio_b64"])
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            previous_text = payload.get("previous_text", "")
            # 2. Inference
            loop = asyncio.get_running_loop()
            new_text = await loop.run_in_executor(
                None, self.run_inference, audio_float32, previous_text, req_id
            )
            latency = round(time.time() - start_time, 3)
            # 3. Handle Result
            if new_text.strip():
                logger.info(
                    f"✅ Result: '{new_text}'",
                    extra={
                        "req_id": req_id,
                        "session_id": session_id,
                        "latency": latency,
                    },
                )

                output_payload = {
                    "req_id": req_id,
                    "session_id": session_id,
                    "text": new_text,
                    "latency": latency,
                    "timestamp": time.time(),
                    "audio_b64": payload["audio_b64"],
                }
                await self.js.publish("asr.output", json.dumps(output_payload).encode())
            else:
                logger.warning(
                    f"🚫 No Result (Silence or Unclear)", extra={"req_id": req_id}
                )

            await msg.ack()

        except Exception as e:
            logger.error(f"❌ Error processing message: {e}", exc_info=True)
            await msg.ack()

    async def start(self):
        self.load_model()
        print(f"🔌 Connecting to NATS: {Config.NATS_URL}")

        try:
            self.nc = await nats.connect(Config.NATS_URL)
            self.js = self.nc.jetstream()

            try:
                await self.js.add_stream(name="ASR_INPUT", subjects=["asr.input"])
            except Exception:
                pass

            await self.js.subscribe(
                "asr.input", queue="asr_workers", cb=self.process_msg, manual_ack=True
            )

            print("🚀 ASR Worker started! Waiting for audio chunks...")
            await asyncio.Future()

        except Exception as e:
            logger.critical(f"Startup failed: {e}")
        finally:
            if self.nc:
                await self.nc.close()


if __name__ == "__main__":
    worker = ASRWorker()
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        print("🛑 Worker stopped.")
