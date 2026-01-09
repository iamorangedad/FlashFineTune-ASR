import asyncio
import json
import base64
import time
import os
import wave
import numpy as np
import nats
from nats.errors import TimeoutError
from faster_whisper import WhisperModel
from src.logger import setup_logger
from src.config import Config

DEVICE = os.getenv("ASR_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float16")
DEBUG_SAVE_AUDIO = os.getenv("DEBUG_SAVE_AUDIO", "true").lower() == "true"

# Use your custom structured logger
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

    def save_debug_wav(self, req_id, audio_int16):
        """Save audio to disk to listen to what the model actually heard"""
        if not DEBUG_SAVE_AUDIO:
            return

        filename = f"/tmp/debug_{req_id}.wav"
        try:
            with wave.open(filename, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_int16.tobytes())
            logger.warning(
                f"💾 Saved debug audio to {filename}", extra={"req_id": req_id}
            )
        except Exception as e:
            logger.error(f"Failed to save debug wav: {e}")

    def run_inference(self, audio_np, previous_text="", req_id="N/A"):
        if not self.model:
            return ""

        # Log Audio Stats to detect silence
        max_amp = np.max(np.abs(audio_np))
        avg_amp = np.mean(np.abs(audio_np))

        # 1. Check for absolute silence
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
            # Run transcription
            # Note: We temporarily relax vad_filter to see if it detects *anything*
            segments_gen, info = self.model.transcribe(
                audio_np,
                beam_size=1,
                language="en",
                initial_prompt=previous_text,
                condition_on_previous_text=True,
                vad_filter=True,  # Keep True, but if it fails often, set False to debug
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Convert generator to list to inspect it
            segments = list(segments_gen)

            if not segments:
                logger.warning(
                    f"⚠️ Whisper finished but found NO segments. (VAD likely filtered it out)",
                    extra={
                        "req_id": req_id,
                        "language_prob": info.language_probability,
                    },
                )
                return ""

            # Log confidence of the first segment
            first_seg_prob = segments[0].avg_logprob
            logger.info(
                f"🧠 Whisper detected language '{info.language}' ({info.language_probability:.2f})",
                extra={"req_id": req_id},
            )

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

            # logger.info(f"Start inference", extra={"req_id": req_id})
            start_time = time.time()

            # 1. 解码音频 (Base64 -> Int16 -> Float32)
            audio_bytes = base64.b64decode(payload["audio_b64"])
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            # Critical: Normalize correctly
            # Float32 must be between -1.0 and 1.0
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # 2. Get Context
            previous_text = payload.get("previous_text", "")

            # 3. Execute Inference in ThreadPool
            loop = asyncio.get_running_loop()
            new_text = await loop.run_in_executor(
                None, self.run_inference, audio_float32, previous_text, req_id
            )

            latency = round(time.time() - start_time, 3)

            # 4. Handle Result
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
                # SAVE THE AUDIO TO DEBUG
                # Run in executor to avoid blocking loop with file I/O
                await loop.run_in_executor(
                    None, self.save_debug_wav, req_id, audio_int16
                )

            await msg.ack()

        except Exception as e:
            logger.error(f"❌ Error processing message: {e}", exc_info=True)
            await msg.ack()

    async def start(self):
        self.load_model()
        logger.info(f"🔌 Connecting to NATS: {Config.NATS_URL}")

        try:
            self.nc = await nats.connect(Config.NATS_URL)
            self.js = self.nc.jetstream()

            # Ensure stream exists
            try:
                await self.js.add_stream(name="ASR_INPUT", subjects=["asr.input"])
            except Exception:
                pass

            await self.js.subscribe(
                "asr.input", queue="asr_workers", cb=self.process_msg, manual_ack=True
            )

            print("🚀 ASR Worker started! Waiting for audio chunks...")
            await asyncio.Future()  # Keep alive

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
