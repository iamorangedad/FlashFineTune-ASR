import gradio as gr
import numpy as np
import websocket
import json
import threading
import queue
import time
import wave  # <--- Added
import uuid  # <--- Added
from scipy import signal
import os
from config import Config

WS_URL = getattr(Config, "WS_URL", "ws://10.0.0.27:30081/ws/realtime")


# --- DEBUG HELPER ---
def save_debug_wav(audio_int16, label):
    """Saves audio chunk to /tmp for debugging"""
    try:
        # Generate a unique filename
        filename = f"/tmp/debug_1_webui_{label}_{uuid.uuid4().hex[:4]}.wav"
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
        print(f"💾 [DEBUG] Saved WebUI audio: {filename}")
    except Exception as e:
        print(f"❌ Failed to save debug wav: {e}")


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

        # --- 1. Stereo to Mono ---
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # --- 2. Resample (to 16000) ---
        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = signal.resample(data, num_samples)

        # --- 3. Convert to Int16 ---
        data_int16 = (data * 32767).astype(np.int16)

        # --- DEBUG SAVE POINT 1: Before Network Send ---
        # Save every chunk to verify local mic input is good
        save_debug_wav(data_int16, "sent")
        # -----------------------------------------------

        try:
            self.ws.send_binary(data_int16.tobytes())
        except Exception as e:
            print(f"Send error: {e}")
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
    except Exception:
        pass

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
