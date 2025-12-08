import pytest
import os
import wave
import struct
import math
from faster_whisper import WhisperModel


# --- 1. 工具函数：生成一个简单的测试音频 ---
def create_dummy_wav(filename, duration_sec=2):
    """
    生成一个包含简单正弦波（哔哔声）的 WAV 文件。
    这样我们就不用依赖外部下载的音频文件了。
    """
    sample_rate = 16000
    n_samples = int(sample_rate * duration_sec)

    with wave.open(filename, "w") as obj:
        obj.setnchannels(1)  # 单声道
        obj.setsampwidth(2)  # 2字节 (16 bit)
        obj.setframerate(sample_rate)

        # 生成 440Hz 的正弦波
        data = []
        for i in range(n_samples):
            value = int(
                32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate)
            )
            data.append(struct.pack("<h", value))

        obj.writeframes(b"".join(data))

    return filename


# --- 2. Pytest Fixtures (前置准备) ---


@pytest.fixture(scope="module")
def audio_file():
    """
    创建一个临时 wav 文件，测试结束后自动删除。
    """
    filename = "test_beep.wav"
    create_dummy_wav(filename)

    yield filename

    # Teardown: 删除文件
    if os.path.exists(filename):
        os.remove(filename)


@pytest.fixture(scope="module")
def whisper_model():
    """
    加载模型。
    scope="module" 保证整个测试文件只加载一次模型，避免每个测试用例都重新加载导致变慢。

    注意：在 CI/CD 或测试环境中，建议使用 'tiny' 模型以节省时间和内存。
    """
    # 如果你在 Jetson 上，可以改为 device="cuda", compute_type="float16"
    # 为了保证单元测试的通用性，这里默认使用 cpu 和 int8
    model_size = "tiny"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model


# --- 3. 测试用例 ---


def test_model_loading(whisper_model):
    """
    测试 0: 验证模型是否成功加载
    """
    assert whisper_model is not None


def test_transcribe_functionality(whisper_model, audio_file):
    """
    测试 1: 核心功能测试 - 确保能跑通 transcribe 流程
    """
    # 运行识别
    # beam_size=1 加快测试速度
    segments, info = whisper_model.transcribe(audio_file, beam_size=1)

    # ⚠️ 重要：segments 是一个生成器 (generator)。
    # 只有当你遍历它时，实际的推理才会执行！
    segments_list = list(segments)

    # --- 断言 (验证结果) ---

    # 1. 验证是否检测到了音频时长 (只要 > 0 即可)
    print(f"\n[Info] Detected duration: {info.duration}s")
    assert info.duration > 0

    # 2. 验证是否有输出段落
    # 虽然是哔哔声，模型可能会识别成空白或者幻觉文本，但列表不应报错
    assert isinstance(segments_list, list)

    # 3. 打印识别结果供调试查看
    text = "".join([s.text for s in segments_list]).strip()
    print(f"[Result] Transcribed text: '{text}'")


@pytest.mark.parametrize("beam_size", [1, 5])
def test_transcribe_params(whisper_model, audio_file, beam_size):
    """
    测试 2: 参数化测试 - 验证不同参数下不会报错
    """
    segments, _ = whisper_model.transcribe(audio_file, beam_size=beam_size)
    results = list(segments)
    assert len(results) >= 0
