#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void compute_mean_var_kernel(const float* input, float* mean, float* var, int N, int C, int H, int W) {
    int c = blockIdx.x;
    int HW = H * W;
    int sampleSize = N * HW;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int n = 0; n < N; ++n) {
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
    var[c] = var_val;
}

__global__ void batchnorm_forward_kernel(const float* input, float* output, const float* mean, const float* var,
                                         const float* gamma, const float* beta,
                                         int N, int C, int H, int W, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (H * W)) % C;
    int n = idx / (C * H * W);

    int offset = ((n * C + c) * H + h) * W + w;

    float inp = input[offset];
    float norm = (inp - mean[c]) * rsqrtf(var[c] + epsilon);
    float out_val = gamma[c] * norm + beta[c];

    output[offset] = out_val;
}

void batchnorm_forward(float* input, float* output, float* mean, float* var,
                       float* gamma, float* beta,
                       int N, int C, int H, int W, float epsilon) {
    int total_size = N * C * H * W;
    int gridSize = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_mean_var_kernel<<<C, 1>>>(input, mean, var, N, C, H, W);
    batchnorm_forward_kernel<<<gridSize, BLOCK_SIZE>>>(
        input, output, mean, var, gamma, beta, N, C, H, W, epsilon
    );

    cudaDeviceSynchronize();
}
