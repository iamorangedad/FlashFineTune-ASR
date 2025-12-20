import time
import nvtx
from faster_whisper import WhisperModel

# 模拟一个音频文件路径，请替换为你真实的音频文件
AUDIO_FILE = "/home/FlashFineTune-ASR/tests/california.mp3"


def main():
    # ---------------------------------------------------------
    # 阶段 1: 模型加载 (Model Loading)
    # 使用蓝色标记，方便在时间轴上快速找到初始化阶段
    # ---------------------------------------------------------
    with nvtx.annotate("Model Loading", color="blue"):
        print("正在加载模型...")
        # 建议：在分析推理性能时，使用 int8 可以获得更真实的 Jetson 性能表现
        model = WhisperModel("small", device="cuda", compute_type="int8")

    # ---------------------------------------------------------
    # 阶段 2: 预处理与准备 (Transcribe Setup)
    # 注意：调用 model.transcribe 只是创建生成器，几乎瞬间完成，还没开始推理！
    # ---------------------------------------------------------
    with nvtx.annotate("Transcribe Setup", color="yellow"):
        print("准备转录...")
        # beam_size=1 通常更快，适合实时性要求高的场景
        segments, info = model.transcribe(AUDIO_FILE, beam_size=5)
        print(f"检测到语言: {info.language}, 概率: {info.language_probability}")

    # ---------------------------------------------------------
    # 阶段 3: 实际推理循环 (Inference Loop)
    # 真正的 GPU 计算发生在这里。我们用绿色标记整个循环。
    # ---------------------------------------------------------
    print("开始推理...")
    start_time = time.time()

    # 这是一个总的范围标记
    with nvtx.annotate("Full Inference Loop", color="green"):

        # 迭代生成器
        for i, segment in enumerate(segments):
            # -----------------------------------------------------
            # 阶段 3.1: 单个片段处理 (Segment Processing)
            # 这个标记内部包含了：
            #   1. VAD (语音活动检测)
            #   2. Encoder 编码
            #   3. Decoder 解码 (生成文本)
            # -----------------------------------------------------
            with nvtx.annotate(f"Segment {i}", color="red"):
                # 这里可以加一些简单的打印，模拟业务逻辑
                # 注意：print 本身是慢操作，正式性能测试时建议去掉
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

                # 如果你有额外的 Python 业务逻辑（比如敏感词过滤），也可以单独包一层
                # with nvtx.annotate("Text Processing", color="white"):
                #     process_text(segment.text)

    total_time = time.time() - start_time
    print(f"转录完成，总耗时: {total_time:.2f}s")


if __name__ == "__main__":
    # 而在最外层加一个标记，方便把整个程序运行时间和系统启动杂项区分开
    with nvtx.annotate("Main App", color="purple"):
        main()
