import gradio as gr
import numpy as np
import websocket
import json
import threading
import queue
import time
from scipy import signal
import os
from config import Config

WS_URL = getattr(Config, "WS_URL", "ws://10.0.0.27:30081/ws/realtime")


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
                # 设置超时以便线程能响应关闭信号
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

        # --- 1. 立体声转单声道 (关键) ---
        # Gradio 有时会给 (N, 2) 的数据
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # --- 2. 重采样 (44100/48000 -> 16000) ---
        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            # resample 返回的是 float64
            data = signal.resample(data, num_samples)

        # --- 3. 类型转换 (Float -> Int16) ---
        # 确保数据在 -1.0 到 1.0 之间
        max_val = np.abs(data).max()
        if max_val > 0:
            # 简单的归一化，防止爆音 (可选)
            # data = data / max_val
            pass

        # 转换为 Int16 PCM
        data_int16 = (data * 32767).astype(np.int16)

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


# --- Session 管理 ---
clients = {}


def get_client(session_hash):
    if session_hash not in clients:
        clients[session_hash] = RealtimeClient()
    return clients[session_hash]


def process_stream(audio, current_text, request: gr.Request):
    if audio is None:
        return current_text, "Ready"

    client = get_client(request.session_hash)

    # 确保连接
    if not client.connected:
        client.connect()

    sr, y = audio

    # 1. 发送音频数据
    client.send_audio_chunk(sr, y)

    # 2. 处理接收队列 (非阻塞)
    try:
        while not client.recv_queue.empty():
            msg = client.recv_queue.get_nowait()

            if msg.get("type") == "update":
                # 拼接文本
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
    """当停止录音或关闭页面时触发"""
    uid = request.session_hash
    if uid in clients:
        clients[uid].close()
        # 这里不一定要 del，因为用户可能马上又要录，保持连接池也可以
        # del clients[uid]


# --- UI 构建 ---
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

    # 事件绑定
    stream_event = input_audio.stream(
        fn=process_stream,
        inputs=[input_audio, output_display],
        outputs=[output_display, latency_display],
        show_progress=False,
    )

    # 停止录音时断开连接 (可选，或者保持连接)
    # input_audio.stop_recording(fn=on_stop)

    # 清除按钮
    clear_btn.click(fn=on_clear, inputs=[], outputs=[output_display, latency_display])

if __name__ == "__main__":
    # 允许局域网访问
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
