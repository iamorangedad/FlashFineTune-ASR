// 假设 blockDim.x = 256
__global__ void log_clamp_reduce_kernel(const float* __restrict__ input, 
                                        float* __restrict__ output, 
                                        float* __restrict__ partial_maxs, 
                                        int n) {
    // 1. 全局索引
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 线程局部变量
    float val = -1e20f; // 初始化为极小值
    
    if (idx < n) {
        float in_val = input[idx];
        // 融合 Clamp 和 Log 操作
        // 对应: torch.clamp(mel_spec, min=1e-10).log10()
        if (in_val < 1e-10f) in_val = 1e-10f;
        val = log10f(in_val);
        
        // 写入中间结果，供下一步使用
        output[idx] = val;
    }

    // 3. Block 内归约 (Block Reduction) 找最大值
    // 使用 Shared Memory
    __shared__ float sdata[256];
    sdata[tid] = val;
    __syncthreads();

    // 经典的树状归约 (Tree Reduction)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 4. 将每个 Block 的最大值写入全局内存
    if (tid == 0) {
        partial_maxs[blockIdx.x] = sdata[0];
    }
}