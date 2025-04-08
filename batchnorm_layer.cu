// batchnorm_layer.cu - Fully optimized CUDA BatchNorm layer (forward only)
#include "batchnorm_layer.h"
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256  // Fixed block size

// Kernel to compute mean per channel
__global__ void compute_mean_kernel(const float* input, float* mean, int N, int C, int H, int W) {
    int c = blockIdx.x;
    int HW = H * W;
    int sampleSize = N * HW;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
        for (int i = 0; i < HW; ++i)
            sum += input[((n * C + c) * HW) + i];

    mean[c] = sum / sampleSize;
}

// Kernel to compute variance per channel
__global__ void compute_variance_kernel(const float* input, const float* mean, float* var, int N, int C, int H, int W) {
    int c = blockIdx.x;
    int HW = H * W;
    int sampleSize = N * HW;

    float m = mean[c];
    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
        for (int i = 0; i < HW; ++i) {
            float val = input[((n * C + c) * HW) + i];
            float diff = val - m;
            sum += diff * diff;
        }

    var[c] = sum / sampleSize;
}

// BatchNorm Forward Pass Kernel (no branching)
__global__ void batchnorm_forward_kernel(float* input, const float* mean, const float* var,
                                         const float* gamma, const float* beta,
                                         int N, int C, int H, int W, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;

    if (idx >= total) return;  // one single exit condition

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (H * W)) % C;
    int n = idx / (C * H * W);
    int offset = ((n * C + c) * H + h) * W + w;

    float m = mean[c];
    float v = var[c];
    float norm = (input[offset] - m) * rsqrtf(v + epsilon);  // rsqrtf is fast and branchless

    input[offset] = norm * gamma[c] + beta[c];
}

// Public functions
void batchnorm_forward(float* input, float* mean, float* var,
                       float* gamma, float* beta,
                       int N, int C, int H, int W, float epsilon) {
    int spatial = N * C * H * W;
    int gridSize = (spatial + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Compute mean and variance per channel
    compute_mean_kernel<<<C, 1>>>(input, mean, N, C, H, W);
    compute_variance_kernel<<<C, 1>>>(input, mean, var, N, C, H, W);

    // Normalize and apply gamma/beta
    batchnorm_forward_kernel<<<gridSize, BLOCK_SIZE>>>(input, mean, var, gamma, beta, N, C, H, W, epsilon);
    cudaDeviceSynchronize();
}
// batchnorm_layer.cu - Fully optimized CUDA BatchNorm layer (forward only)
#include "batchnorm_layer.h"
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256  // Fixed block size

// Kernel to compute mean per channel
__global__ void compute_mean_kernel(const float* input, float* mean, int N, int C, int H, int W) {
    int c = blockIdx.x;
    int HW = H * W;
    int sampleSize = N * HW;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
        for (int i = 0; i < HW; ++i)
            sum += input[((n * C + c) * HW) + i];

    mean[c] = sum / sampleSize;
}

// Kernel to compute variance per channel
__global__ void compute_variance_kernel(const float* input, const float* mean, float* var, int N, int C, int H, int W) {
    int c = blockIdx.x;
    int HW = H * W;
    int sampleSize = N * HW;

    float m = mean[c];
    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
        for (int i = 0; i < HW; ++i) {
            float val = input[((n * C + c) * HW) + i];
            float diff = val - m;
            sum += diff * diff;
        }

    var[c] = sum / sampleSize;
}

// BatchNorm Forward Pass Kernel (no branching)
__global__ void batchnorm_forward_kernel(float* input, const float* mean, const float* var,
                                         const float* gamma, const float* beta,
                                         int N, int C, int H, int W, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;

    if (idx >= total) return;  // one single exit condition

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (H * W)) % C;
    int n = idx / (C * H * W);
    int offset = ((n * C + c) * H + h) * W + w;

    float m = mean[c];
    float v = var[c];
    float norm = (input[offset] - m) * rsqrtf(v + epsilon);  // rsqrtf is fast and branchless

    input[offset] = norm * gamma[c] + beta[c];
}

// Public functions
void batchnorm_forward(float* input, float* mean, float* var,
                       float* gamma, float* beta,
                       int N, int C, int H, int W, float epsilon) {
    int spatial = N * C * H * W;
    int gridSize = (spatial + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Compute mean and variance per channel
    compute_mean_kernel<<<C, 1>>>(input, mean, N, C, H, W);
    compute_variance_kernel<<<C, 1>>>(input, mean, var, N, C, H, W);

    // Normalize and apply gamma/beta
    batchnorm_forward_kernel<<<gridSize, BLOCK_SIZE>>>(input, mean, var, gamma, beta, N, C, H, W, epsilon);
    cudaDeviceSynchronize();
}
