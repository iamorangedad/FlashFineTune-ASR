#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ==========================================
// Part 1: Device Code (customed GPU Kernel)
// ==========================================
__global__ void log_clamp_reduce_kernel(const float* __restrict__ input, 
                                        float* __restrict__ output, 
                                        float* __restrict__ partial_maxs, 
                                        int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    // init the minmum value
    float val = -1e20f; 
    if (idx < n) {
        float in_val = input[idx];
        if (in_val < 1e-10f) in_val = 1e-10f;
        val = log10f(in_val);
        output[idx] = val;
    }
    // Shared Memory is matched with blockDim.x
    __shared__ float sdata[256];
    sdata[tid] = val;
    __syncthreads();
    // Tree Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_maxs[blockIdx.x] = sdata[0];
    }
}

__global__ void normalize_kernel(float* __restrict__ data, int n, float global_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        float limit = global_max - 8.0f;
        val = fmaxf(val, limit);
        val = (val + 4.0f) * 0.25f;
        data[idx] = val;
    }
}

// ==========================================
// Part 2: Host Code (C++ wrapper)
// ==========================================
torch::Tensor compute_log_mel_cuda(torch::Tensor input) {
    // 1. Input Checks
    auto input_cont = input.contiguous(); 
    int n = input_cont.numel();
    
    // 2. Allocate Output
    auto output = torch::empty_like(input_cont);
    
    // 3. Kerne Config
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    // apply temporary space for partial max (就在显存上申请一个小的 Tensor)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto partial_maxs = torch::empty({blocks}, options);

    // 4. Launch Kernel 1
    log_clamp_reduce_kernel<<<blocks, threads>>>(
        input_cont.data_ptr<float>(),
        output.data_ptr<float>(),
        partial_maxs.data_ptr<float>(),
        n
    );
    
    // 5. Max
    float global_max = partial_maxs.max().item<float>();

    // 6. Launch Kernel 2
    normalize_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        n,
        global_max
    );
    return output;
}

// ==========================================
// Part 3: Pybind11 binding
// ==========================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_log_mel", &compute_log_mel_cuda, "Log Mel Spectrogram CUDA");
}