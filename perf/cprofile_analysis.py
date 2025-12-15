import cProfile
import pstats
from faster_whisper import WhisperModel


# 你的业务函数
def my_transcription_task():
    model = WhisperModel("tiny", device="cuda", compute_type="float16")

    # 注意：必须消耗生成器才能触发实际计算
    segments, info = model.transcribe("california.mp3", beam_size=5)

    # 强制将生成器转换为列表，以此触发完整的推理过程
    result = list(segments)
    print(f"Detected language: {info.language}")


if __name__ == "__main__":
    # 创建 Profiler 对象
    profiler = cProfile.Profile()

    # 开始抓取
    profiler.enable()
    my_transcription_task()
    profiler.disable()

    # 保存结果到文件
    profiler.dump_stats("whisper_analysis.prof")
    print("性能分析完成，结果已保存为 whisper_analysis.prof")
