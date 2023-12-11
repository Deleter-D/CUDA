#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

void sumArraysHost(float *A, float *B, float *C, const int size)
{
    for (int i = 0; i < size; i++)
        C[i] = A[i] + B[i];
}

__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
        C[tid] = A[tid] + B[tid];
}

int main(int argc, char const *argv[])
{
    int device = 0;
    setDevice(device);

    // 检查设备是否支持统一虚拟地址空间
    cudaDeviceProp prop;
    ERROR_CHECK(cudaGetDeviceProperties(&prop, device));
    if (!prop.unifiedAddressing)
    {
        printf("Device %d does not support unified virtual address space!\n", device);
        ERROR_CHECK(cudaDeviceReset());
        return 1;
    }

    int power = 10;
    if (argc > 1)
        power = atoi(argv[1]);
    int size     = 1 << power;
    size_t bytes = size * sizeof(float);
    if (power < 18)
        printf("Vector size %d,\t%3.0f KB\n", size, (float)bytes / 1024.0f);
    else
        printf("Vector size %d,\t%3.0f MB\n", size, (float)bytes / (1024.0f * 1024.0f));

    dim3 block(512);
    dim3 grid((size + block.x - 1) / block.x);

    // 使用统一虚拟地址空间
    float *h_A, *h_B, *hostRef, *gpuRef;
    ERROR_CHECK(cudaHostAlloc((void **)&h_A, bytes, cudaHostAllocMapped));
    ERROR_CHECK(cudaHostAlloc((void **)&h_B, bytes, cudaHostAllocMapped));
    hostRef = (float *)malloc(bytes);
    gpuRef  = (float *)malloc(bytes);

    initializeData<float>(h_A, size);
    initializeData<float>(h_B, size);
    memset(hostRef, 0, bytes);
    memset(gpuRef, 0, bytes);

    float *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    // 通过cudaPointerGetAttributes来获取指针属性
    // cudaPointerAttributes attr;
    // ERROR_CHECK(cudaPointerGetAttributes(&attr, h_A));
    // int dev = attr.device;             // 内存所在的设备
    // auto host_ptr = attr.hostPointer;  // 获取主机端的指针，若没有则为空，可以通过&运算符操作
    // auto dev_ptr = attr.devicePointer; // 获取设备端的指针，若没有则为空，可以通过&运算符操作
    // cudaMemoryType type = attr.type;   // 获取内存类型

    sumArraysHost(h_A, h_B, hostRef, size);
    sumArraysZeroCopy<<<grid, block>>>(h_A, h_B, d_C, size);

    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());

    checkResult<float>(hostRef, gpuRef, size);

    ERROR_CHECK(cudaFree(d_C));
    ERROR_CHECK(cudaFreeHost(h_A));
    ERROR_CHECK(cudaFreeHost(h_B));

    free(hostRef);
    free(gpuRef);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}