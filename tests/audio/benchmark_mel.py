import torch
import math
import time
from torch.utils.cpp_extension import load_inline

# =========================================================
# Part 1: CUDA C++ Source Code
# =========================================================
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

// Kernel 1: Log10, Clamp, and Block-level Reduction (Find Max)
// 输入: input (magnitudes from mel filter)
// 输出: output (log_spec), partial_maxs (per-block max values)
__global__ void log_clamp_reduce_kernel(const float* __restrict__ input, 
                                        float* __restrict__ output, 
                                        float* __restrict__ partial_maxs, 
                                        int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = -1e20f; // 初始化极小值

    // 1. 融合操作: Clamp -> Log10
    if (idx < n) {
        float in_val = input[idx];
        // torch.clamp(x, min=1e-10)
        if (in_val < 1e-10f) in_val = 1e-10f;
        // .log10()
        val = log10f(in_val);
        // 写入中间结果
        output[idx] = val;
    }

    // 2. Block 内归约 (Shared Memory Reduction)
    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = val;
    __syncthreads();

    // 树状归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 每个 Block 的线程 0 将该 Block 的最大值写入全局内存
    if (tid == 0) {
        partial_maxs[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Normalization
// 输入: log_spec (from Kernel 1), global_max (computed scalar)
// 输出: 原地修改 log_spec
__global__ void normalize_kernel(float* __restrict__ data, 
                                 int n, 
                                 float global_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // torch.maximum(log_spec, log_spec.max() - 8.0)
        float limit = global_max - 8.0f;
        val = fmaxf(val, limit);
        
        // (val + 4.0) / 4.0
        val = (val + 4.0f) * 0.25f;
        
        data[idx] = val;
    }
}

// C++ Launcher Code
torch::Tensor optimized_log_mel(torch::Tensor input) {
    auto n = input.numel();
    auto output = torch::empty_like(input);
    
    // 配置 Grid 和 Block
    int threads = BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    
    // 申请临时空间存放每个 Block 的最大值
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto partial_maxs = torch::empty({blocks}, options);

    // 启动 Kernel 1: Log + Clamp + Partial Reduce
    log_clamp_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        partial_maxs.data_ptr<float>(),
        n
    );
    
    // 快速计算全局最大值
    // 这里偷个懒：由于 partial_maxs 非常小 (e.g. 100个元素), 
    // 让 PyTorch 调度一个小 kernel 归约比我们手写 global atomic 更简单且性能损失忽略不计
    float global_max = partial_maxs.max().item<float>();

    // 启动 Kernel 2: Normalize
    normalize_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        n,
        global_max
    );

    return output;
}
"""

cpp_source = r"""
torch::Tensor optimized_log_mel(torch::Tensor input);
"""

# 编译并加载 CUDA 扩展
print("Compiling CUDA extension... (This takes a few seconds the first time)")
fused_ops = load_inline(
    name="fused_mel_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["optimized_log_mel"],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
)


# =========================================================
# Part 2: Original PyTorch Implementation
# =========================================================
def original_log_mel(mel_spec_input):
    # 输入已经是 filter @ magnitudes 之后的结果
    log_spec = torch.clamp(mel_spec_input, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


# =========================================================
# Part 3: Correctness & Performance Test
# =========================================================
def run_benchmark():
    # 1. 模拟数据: [Batch=1, Mel=80, Frames=3000] (约 30秒音频)
    # 在 Jetson 上显存是共享的，但尽量用 CUDA tensor
    device = torch.device("cuda")
    B, M, T = 1, 80, 3000
    # 模拟 filters @ magnitudes 的结果
    mel_spec = torch.rand((B, M, T), device=device, dtype=torch.float32) * 100.0

    print(f"Data Shape: {mel_spec.shape}, Device: {device}")

    # --- 正确性验证 ---
    print("\nVerifying Correctness...")
    res_torch = original_log_mel(mel_spec)
    res_cuda = fused_ops.optimized_log_mel(mel_spec)

    # 允许一点点浮点误差
    if torch.allclose(res_torch, res_cuda, atol=1e-5):
        print("✅ Results Match! The CUDA kernel is correct.")
    else:
        print("❌ Results Do Not Match!")
        print("Max Diff:", (res_torch - res_cuda).abs().max().item())
        return

    # --- 性能测试 ---
    print("\nStarting Benchmark (1000 runs)...")

    # Warmup
    for _ in range(100):
        original_log_mel(mel_spec)
        fused_ops.optimized_log_mel(mel_spec)

    torch.cuda.synchronize()

    # Test Original
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(1000):
        original_log_mel(mel_spec)
    end.record()
    torch.cuda.synchronize()
    time_torch = start.elapsed_time(end) / 1000.0  # ms

    # Test Custom CUDA
    start.record()
    for _ in range(1000):
        fused_ops.optimized_log_mel(mel_spec)
    end.record()
    torch.cuda.synchronize()
    time_cuda = start.elapsed_time(end) / 1000.0  # ms

    print(f"{'Implementation':<20} | {'Avg Latency (ms)':<15} | {'FPS (Approx)':<15}")
    print("-" * 60)
    print(
        f"{'Original PyTorch':<20} | {time_torch:.4f} ms        | {1000/time_torch:.1f}"
    )
    print(f"{'Custom CUDA':<20} | {time_cuda:.4f} ms        | {1000/time_cuda:.1f}")
    print("-" * 60)
    print(f"🚀 Speedup: {time_torch / time_cuda:.2f}x")


if __name__ == "__main__":
    run_benchmark()
