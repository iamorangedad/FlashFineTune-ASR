import torch

# 导入你编译好的 C++ 扩展
import fast_mel_cuda_impl


class AudioPreprocessor:
    def __init__(self):
        pass

    def process(self, audio_tensor):
        """
        输入: audio_tensor (CUDA FloatTensor)
        输出: log_mel (CUDA FloatTensor)
        """
        # 你的 C++ 函数已经封装好了所有逻辑
        # 注意：这里已经是直接在 GPU 上极速运行了
        return fast_mel_cuda_impl.compute_log_mel(audio_tensor)


# --- 测试调用 ---
if __name__ == "__main__":
    device = "cuda"
    # 模拟输入
    mel_input = torch.rand((1, 80, 3000), device=device)

    processor = AudioPreprocessor()

    # 第一次运行
    output = processor.process(mel_input)
    print(f"Output shape: {output.shape}")
