#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

// Forward declaration of batchnorm forward pass
void batchnorm_forward(float* input, float* mean, float* var,
                       float* gamma, float* beta,
                       int N, int C, int H, int W, float epsilon);

#endif
