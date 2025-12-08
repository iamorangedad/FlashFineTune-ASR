FROM python:3.10-slim

WORKDIR /app
# 安装 ffmpeg (如果 whisper 需要)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# 默认指令不写，在 K8s yaml 中指定