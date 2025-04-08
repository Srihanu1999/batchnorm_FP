#include <iostream>
#include <cuda_runtime.h>
#include "batchnorm_layer.h"

void print_array(const float* arr, int N) {
    for (int i = 0; i < N; ++i)
        std::cout << arr[i] << " ";
    std::cout << "\n";
}

int main() {
    const int N = 1, C = 2, H = 2, W = 2;
    const int size = N * C * H * W;
    const float epsilon = 1e-5;

    float h_input[size] = {1, 2, 3, 4, 5, 6, 7, 8}; // Simple tensor
    float h_gamma[C] = {1.0f, 1.0f};  // No scaling
    float h_beta[C] = {0.0f, 0.0f};   // No shifting

    // Allocate device memory
    float *d_input, *d_mean, *d_var, *d_gamma, *d_beta;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_mean, C * sizeof(float));
    cudaMalloc(&d_var, C * sizeof(float));
    cudaMalloc(&d_gamma, C * sizeof(float));
    cudaMalloc(&d_beta, C * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice);

    // Run forward pass
    batchnorm_forward(d_input, d_mean, d_var, d_gamma, d_beta, N, C, H, W, epsilon);

    // Copy result back
    float h_output[size];
    cudaMemcpy(h_output, d_input, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "BatchNorm Output:\n";
    print_array(h_output, size);

    // Optional: Print mean and variance
    float h_mean[C], h_var[C];
    cudaMemcpy(h_mean, d_mean, C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_var, d_var, C * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Mean: ";
    print_array(h_mean, C);
    std::cout << "Variance: ";
    print_array(h_var, C);

    // Clean up
    cudaFree(d_input); cudaFree(d_mean); cudaFree(d_var);
    cudaFree(d_gamma); cudaFree(d_beta);

    return 0;
}
