#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

// Compute mean and variance in a single kernel (no branching)
__global__ void compute_mean_var_kernel(const float* input, float* mean, float* var, int N, int C, int H, int W) {
    int c = blockIdx.x;  // One block per channel
    int HW = H * W;
    int sampleSize = N * HW;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    #pragma unroll
    for (int n = 0; n < N; ++n) {
        #pragma unroll
        for (int i = 0; i < HW; ++i) {
            int idx = ((n * C + c) * HW) + i;
            float val = input[idx];
            sum += val;
            sum_sq += val * val;
        }
    }

    float mean_val = sum / sampleSize;
    float var_val = (sum_sq / sampleSize) - mean_val * mean_val;

    mean[c] = mean_val;
    var[c]  = var_val;
}

// Forward pass kernel without any branching
__global__ void batchnorm_forward_kernel(float* input, const float* mean, const float* var,
                                         const float* gamma, const float* beta,
                                         int N, int C, int H, int W, float epsilon, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Always run, even if idx >= total_size. We'll use ternary-like math trick to suppress out-of-bounds.
    int valid = (idx < total_size);            // 1 if valid, 0 if invalid
    int safe_idx = idx * valid;                // valid idx, or 0 if invalid

    int w = safe_idx % W;
    int h = (safe_idx / W) % H;
    int c = (safe_idx / (H * W)) % C;
    int n = safe_idx / (C * H * W);

    int offset = ((n * C + c) * H + h) * W + w;

    float inp = input[offset * valid];         // 0 if invalid
    float norm = (inp - mean[c]) * rsqrtf(var[c] + epsilon);
    float out_val = gamma[c] * norm + beta[c];

    input[offset * valid] = out_val * valid + input[offset * (1 - valid)];  // no if, always write
}

// Entry function (no branches)
void batchnorm_forward(float* input, float* mean, float* var,
                       float* gamma, float* beta,
                       int N, int C, int H, int W, float epsilon) {
    int total_size = N * C * H * W;
    int gridSize = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_mean_var_kernel<<<C, 1>>>(input, mean, var, N, C, H, W);
    batchnorm_forward_kernel<<<gridSize, BLOCK_SIZE>>>(
        input, mean, var, gamma, beta, N, C, H, W, epsilon, total_size
    );

    cudaDeviceSynchronize();
}

