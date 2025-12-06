# worker.py
import asyncio
import json
import nats
import boto3
import os

s3_client = boto3.client("s3")


def upload_to_s3(request_id, audio_bytes):
    s3_key = f"raw_audio/{request_id}.wav"
    s3_client.put_object(Bucket="asr-logs", Key=s3_key, Body=audio_bytes)


def save_to_mongo(request_id, s3_key, transcript, meta):

    log_entry = {
        "request_id": request_id,
        "s3_path": s3_key,
        "transcript": transcript,
        "meta": meta,
    }

    with open("asr_session_logs.jsonl", "a", encoding="utf-8") as f:

        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


async def main():
    # 1. 连接 NATS
    nc = await nats.connect("nats://nats:4222")
    js = nc.jetstream()

    # 2. 确保 Stream 存在 (幂等操作)
    # 定义一个名为 ASR_LOGS 的流，监听 asr.logs.* 主题
    try:
        await js.add_stream(name="ASR_LOGS", subjects=["asr.logs.*"])
    except Exception as e:
        print(f"Stream ensure error (ignore if exists): {e}")

    # 3. 创建持久化订阅 (Durable Consumer)
    # 即使 Worker 重启，也能从上次断开的地方继续消费
    psub = await js.pull_subscribe("asr.logs.new", durable="logger_worker_v1")

    print("Worker started. Waiting for messages...")

    while True:
        try:
            # 拉取一批消息
            msgs = await psub.fetch(1, timeout=5)
            for msg in msgs:
                data = json.loads(msg.data.decode())
                print(f"Processing log: {data['request_id']}")

                # --- 这里执行耗时操作 ---
                upload_to_s3(request_id, data["audio"])
                save_to_mongo(data["meta"])
                # -----------------------

                # 关键：处理成功后发送确认 (Ack)
                # 如果这里崩了没 Ack，NATS 会在超时后自动重发给其他 Worker
                await msg.ack()

        except nats.errors.TimeoutError:
            # 队列空了，歇会儿
            continue
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
