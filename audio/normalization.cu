__global__ void normalize_kernel(float* __restrict__ data, 
                                 int n, 
                                 float global_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        
        // 对应: torch.maximum(log_spec, log_spec.max() - 8.0)
        float limit = global_max - 8.0f;
        val = fmaxf(val, limit);
        
        // 对应: (log_spec + 4.0) / 4.0
        val = (val + 4.0f) * 0.25f; // 乘法比除法快
        
        data[idx] = val;
    }
}