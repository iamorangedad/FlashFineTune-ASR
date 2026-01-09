import gradio as gr
import numpy as np
import websocket
import json
import threading
import queue
import time
import wave
import uuid
from scipy import signal
from scipy.signal import resample_poly
import os
from config import Config

WS_URL = getattr(Config, "WS_URL", "ws://10.0.0.27:30081/ws/realtime")


# --- DEBUG HELPER ---
def save_debug_wav(audio_int16, label):
    """Saves audio chunk to /tmp for debugging"""
    try:
        filename = f"/tmp/debug_1_webui_{label}_{uuid.uuid4().hex[:4]}.wav"
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
        print(f"💾 [DEBUG] Saved WebUI audio: {filename}")
    except Exception as e:
        print(f"❌ Failed to save debug wav: {e}")


# --- AUDIO NORMALIZATION ---
def validate_and_normalize(data):
    """Convert any audio format to float32 in [-1, 1] range"""
    original_dtype = data.dtype
    original_range = (data.min(), data.max())

    # Convert to float64 for safe calculations
    data = data.astype(np.float64)

    # Handle different input formats
    if original_dtype in [np.int16, np.int32, np.int64]:
        if original_dtype == np.int16:
            data = data / 32768.0
            print(
                f"📊 Converted int16 → float: range {original_range} → [{data.min():.4f}, {data.max():.4f}]"
            )
        elif original_dtype == np.int32:
            data = data / 2147483648.0
            print(
                f"📊 Converted int32 → float: range {original_range} → [{data.min():.4f}, {data.max():.4f}]"
            )
        elif original_dtype == np.int64:
            data = data / 9223372036854775808.0
            print(
                f"📊 Converted int64 → float: range {original_range} → [{data.min():.4f}, {data.max():.4f}]"
            )
    elif original_dtype in [np.uint8, np.uint16]:
        if original_dtype == np.uint8:
            data = (data.astype(np.float64) - 128) / 128.0
            print(
                f"📊 Converted uint8 → float: range {original_range} → [{data.min():.4f}, {data.max():.4f}]"
            )
        elif original_dtype == np.uint16:
            data = (data.astype(np.float64) - 32768) / 32768.0
            print(
                f"📊 Converted uint16 → float: range {original_range} → [{data.min():.4f}, {data.max():.4f}]"
            )
    else:
        # Already float - check range
        if data.max() > 1.0 or data.min() < -1.0:
            peak = np.abs(data).max()
            data = data / peak if peak > 0 else data
            print(
                f"📊 Peak-normalized float: range {original_range} → [{data.min():.4f}, {data.max():.4f}]"
            )
        else:
            print(f"📊 Audio already in range: {original_dtype}{original_range}")

    return data.astype(np.float32)


# --------------------


class RealtimeClient:
    def __init__(self):
        self.ws = None
        self.recv_queue = queue.Queue()
        self.full_text = ""
        self.latency_info = "Latency: N/A"
        self.connected = False
        self.running = False

    def connect(self):
        if self.connected:
            return
        try:
            self.ws = websocket.create_connection(WS_URL, timeout=5)
            self.connected = True
            self.running = True
            threading.Thread(target=self._recv_loop, daemon=True).start()
            print("✅ Websocket connected")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False

    def _recv_loop(self):
        while self.running and self.connected:
            try:
                self.ws.settimeout(1)
                msg = self.ws.recv()
                data = json.loads(msg)
                self.recv_queue.put(data)
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                print(f"Websocket read error: {e}")
                self.connected = False
                break

    def send_audio_chunk(self, sr, data):
        if not self.connected or self.ws is None:
            return

        try:
            print(
                f"\n📥 Received audio chunk from Gradio: sr={sr}Hz, shape={data.shape}, dtype={data.dtype}"
            )

            # --- 1. Validate & Normalize Input ---
            data = validate_and_normalize(data)

            # --- 2. Stereo to Mono ---
            if len(data.shape) > 1:
                print(f"⚠️  Converting stereo {data.shape} to mono")
                data = np.mean(data, axis=1)

            print(f"   Shape after mono conversion: {data.shape}")

            # --- 3. Resample to 16kHz using high-quality method ---
            target_sr = 16000
            if sr != target_sr:
                print(f"🔄 Resampling {sr}Hz → {target_sr}Hz using polyphase filter")
                gcd = np.gcd(sr, target_sr)
                up = target_sr // gcd
                down = sr // gcd
                print(f"   Resampling ratio: {up}/{down}")

                data = resample_poly(data, up, down)

                # Check for clipping artifacts
                peak = np.abs(data).max()
                if peak > 1.05:  # Allow small numerical error
                    print(
                        f"   ⚠️  Resampled peak exceeded 1.0: {peak:.4f}, normalizing..."
                    )
                    data = data / peak

                print(
                    f"   ✅ Resampled successfully, new shape: {data.shape}, peak: {peak:.4f}"
                )
            else:
                print(f"   ✅ Already at target sample rate")

            # --- 4. Convert to Int16 Safely with Clipping ---
            print(f"   Converting to int16...")
            data_int16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)

            peak_int16 = np.abs(data_int16).max()
            print(
                f"   ✅ Int16 conversion complete, peak: {peak_int16}, {len(data_int16)} samples"
            )

            # --- 5. DEBUG SAVE ---
            save_debug_wav(data_int16, f"sent_sr{sr}")

            # --- 6. Send to Server ---
            self.ws.send_binary(data_int16.tobytes())
            print(f"✉️  Sent {len(data_int16) * 2} bytes to server\n")

        except Exception as e:
            print(f"❌ Send error: {e}")
            import traceback

            traceback.print_exc()
            self.connected = False

    def close(self):
        self.running = False
        self.connected = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        print("🔌 Websocket closed")

    def clear_text(self):
        self.full_text = ""
        self.latency_info = "Latency: N/A"


# --- Session Management ---
clients = {}


def get_client(session_hash):
    if session_hash not in clients:
        clients[session_hash] = RealtimeClient()
    return clients[session_hash]


def process_stream(audio, current_text, request: gr.Request):
    if audio is None:
        return current_text, "Ready"

    client = get_client(request.session_hash)

    # Ensure connection
    if not client.connected:
        client.connect()

    sr, y = audio

    # 1. Send Audio
    client.send_audio_chunk(sr, y)

    # 2. Process Receive Queue
    try:
        while not client.recv_queue.empty():
            msg = client.recv_queue.get_nowait()
            if msg.get("type") == "update":
                text_chunk = msg.get("text", "")
                latency = msg.get("latency", 0)
                client.full_text += text_chunk
                client.latency_info = f"Latency: {latency:.3f}s"
    except Exception as e:
        print(f"Error processing queue: {e}")

    return client.full_text, client.latency_info


def on_clear(request: gr.Request):
    client = get_client(request.session_hash)
    client.clear_text()
    return "", "Latency: N/A"


def on_stop(request: gr.Request):
    uid = request.session_hash
    if uid in clients:
        clients[uid].close()


# --- UI Build ---
with gr.Blocks(title="ASR Realtime Client") as demo:
    gr.Markdown("### 🎙️ Distributed ASR Realtime Client")
    gr.Markdown(f"Connecting to: `{WS_URL}`")
    gr.Markdown(
        "📌 **Audio is being validated and normalized. Check console for debug info.**"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_audio = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="Microphone Input",
            )
            clear_btn = gr.Button("Clear Text & Reset")
            latency_display = gr.Label(value="Latency: N/A", label="System Metrics")

        with gr.Column(scale=2):
            output_display = gr.Textbox(
                label="Recognized Text",
                lines=10,
                placeholder="Start speaking...",
                interactive=False,
            )

    stream_event = input_audio.stream(
        fn=process_stream,
        inputs=[input_audio, output_display],
        outputs=[output_display, latency_display],
        show_progress=False,
    )
    clear_btn.click(fn=on_clear, inputs=[], outputs=[output_display, latency_display])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
