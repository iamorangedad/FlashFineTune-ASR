import gradio as gr
import numpy as np
import websocket
import json
import threading
import queue
import time
from scipy import signal  # 用于重采样
from config import Config


class RealtimeClient:
    def __init__(self):
        self.ws = None
        self.recv_queue = queue.Queue()
        self.full_text = ""
        self.connected = False
        self.lock = threading.Lock()

    def connect(self):
        try:
            self.ws = websocket.create_connection(Config.WS_URL)
            self.connected = True
            threading.Thread(target=self._recv_loop, daemon=True).start()
            print("Websocket connected")
        except Exception as e:
            print(f"Connection failed: {e}")

    def _recv_loop(self):
        while self.connected:
            try:
                msg = self.ws.recv()
                data = json.loads(msg)
                self.recv_queue.put(data)
                if data.get("type") == "finish":
                    break
            except:
                break

    def send_audio_chunk(self, sr, data):
        if not self.connected or self.ws is None:
            return

        # --- 音频预处理关键步骤 ---
        # 1. 重采样: Gradio 可能给 44100Hz or 48000Hz, Whisper 需要 16000Hz
        if sr != 16000:
            # 计算重采样后的点数
            num_samples = int(len(data) * 16000 / sr)
            data = signal.resample(data, num_samples)

        # 2. 类型转换: Float32 -> Int16 PCM
        # Gradio 输出通常是 float32 (-1.0 ~ 1.0)
        # 我们转换为 int16 发送给后端以节省网络带宽 (byte流)
        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)

        try:
            self.ws.send_binary(data.tobytes())
        except Exception as e:
            print(f"Send error: {e}")

    def close(self):
        self.connected = False
        if self.ws:
            try:
                self.ws.send("EOF")
                self.ws.close()
            except:
                pass


# Session 管理
clients = {}


def process_stream(audio, current_text, request: gr.Request):
    if audio is None:
        return current_text

    uid = request.session_hash
    if uid not in clients:
        clients[uid] = RealtimeClient()
        clients[uid].connect()

    client = clients[uid]
    sr, y = audio

    # 发送音频
    client.send_audio_chunk(sr, y)

    # 接收文本更新
    try:
        while not client.recv_queue.empty():
            msg = client.recv_queue.get_nowait()
            if msg["type"] == "update":
                # 服务端返回的是增量文本，我们拼接到总文本后
                # 注意：实际生产中可能需要处理重复词，这里简化为直接拼接
                client.full_text += msg["text"]
    except:
        pass

    return client.full_text


def on_stop_recording(request: gr.Request):
    uid = request.session_hash
    if uid in clients:
        clients[uid].close()
        del clients[uid]


with gr.Blocks(title="Whisper Realtime") as demo:
    gr.Markdown("### 🚀 Faster-Whisper 实时流式推理")

    with gr.Row():
        input_audio = gr.Audio(
            sources=["microphone"],
            streaming=True,
            type="numpy",  # 获取原始数据自行处理
            label="Speak Here",
        )
        output_display = gr.Textbox(label="Result", lines=8)

    # 这里的 state 其实没用到，因为我们用 class 管理了状态，但保留以防万一
    state = gr.State()

    input_audio.stream(
        fn=process_stream,
        inputs=[input_audio, output_display],
        outputs=[output_display],
        show_progress=False,
    )

    input_audio.clear(fn=on_stop_recording)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
