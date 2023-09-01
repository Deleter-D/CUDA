#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void VecAdd(float *A, float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(int argc, char const *argv[])
{
    int N = 64;
    size_t size = N * sizeof(float);

    // 申请主机内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 初始化向量
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    // 申请设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Host -> Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 核函数调用
    VecAdd<<<4, 16>>>(d_A, d_B, d_C, N);

    // Device -> Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        std::cout << h_C[i] << " ";
    std::cout << std::endl;

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}