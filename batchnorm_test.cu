#include <iostream>
#include <cuda_runtime.h>
#include "batchnorm_layer.h"

#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
}

void print_tensor(const char* label, float* data, int N, int C, int H, int W) {
    std::cout << label << ":\n";
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            std::cout << "N=" << n << ", C=" << c << ":\n";
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    std::cout << data[idx] << " ";
                }
                std::cout << "\n";
            }
        }
    std::cout << std::endl;
}

int main() {
    const int N = 1, C = 2, H = 2, W = 2;
    const int total = N * C * H * W;
    const float epsilon = 1e-5f;

    float h_input[total] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_gamma[C] = {1.0f, 1.0f};
    float h_beta[C] = {0.0f, 0.0f};

    float *d_input, *d_output, *d_mean, *d_var, *d_gamma, *d_beta;
    CHECK_CUDA(cudaMalloc(&d_input, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mean, C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_var, C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gamma, C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice));

    batchnorm_forward(d_input, d_output, d_mean, d_var, d_gamma, d_beta, N, C, H, W, epsilon);

    float h_output[total], h_mean[C], h_var[C];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mean, d_mean, C * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_var, d_var, C * sizeof(float), cudaMemcpyDeviceToHost));

    print_tensor("Input Normalized", h_output, N, C, H, W);

    std::cout << "Mean: ";
    for (int i = 0; i < C; ++i) std::cout << h_mean[i] << " ";
    std::cout << "\nVariance: ";
    for (int i = 0; i < C; ++i) std::cout << h_var[i] << " ";
    std::cout << std::endl;

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_mean);
    cudaFree(d_var); cudaFree(d_gamma); cudaFree(d_beta);
    return 0;
}
