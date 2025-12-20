#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *hA, *hB, *hC;
    cudaMallocHost(&hA, size);
    cudaMallocHost(&hB, size);
    cudaMallocHost(&hC, size);

    for (int i = 0; i < N; i++) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    vectorAdd<<<(N + 255) / 256, 256>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    printf("C[0] = %f\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);

    return 0;
}
