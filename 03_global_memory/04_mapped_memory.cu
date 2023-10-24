#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析代码
    nsys nvprof /path/to/04_mapped_memory 8
    nsys nvprof /path/to/04_mapped_memory 10
    nsys nvprof /path/to/04_mapped_memory 12
    nsys nvprof /path/to/04_mapped_memory 14
    nsys nvprof /path/to/04_mapped_memory 16
    nsys nvprof /path/to/04_mapped_memory 18
    nsys nvprof /path/to/04_mapped_memory 20
    nsys nvprof /path/to/04_mapped_memory 22
    nsys nvprof /path/to/04_mapped_memory 24
*/

void sumArraysHost(float *A, float *B, float *C, const int size)
{
    for (int i = 0; i < size; i++)
        C[i] = A[i] + B[i];
}

__global__ void sumArrays(float *A, float *B, float *C, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
        C[tid] = A[tid] + B[tid];
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

    // 检查设备是否支持映射内存
    cudaDeviceProp prop;
    ERROR_CHECK(cudaGetDeviceProperties(&prop, device));
    if (!prop.canMapHostMemory)
    {
        printf("Device %d does not support mapping host memory!\n", device);
        ERROR_CHECK(cudaDeviceReset());
        return 1;
    }

    int power = 10;
    if (argc > 1)
        power = atoi(argv[1]);
    int size = 1 << power;
    size_t bytes = size * sizeof(float);
    if (power < 18)
        printf("Vector size %d,\t%3.0f KB\n", size, (float)bytes / 1024.0f);
    else
        printf("Vector size %d,\t%3.0f MB\n", size, (float)bytes / (1024.0f * 1024.0f));

    // 使用设备内存
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    hostRef = (float *)malloc(bytes);
    gpuRef = (float *)malloc(bytes);

    initializeData<float>(h_A, size);
    initializeData<float>(h_B, size);
    memset(hostRef, 0, bytes);
    memset(gpuRef, 0, bytes);

    float *d_A, *d_B, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_B, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(512);
    dim3 grid((size + block.x - 1) / block.x);

    sumArraysHost(h_A, h_B, hostRef, size);
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, size);

    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());

    checkResult<float>(hostRef, gpuRef, size);

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    // 使用映射内存
    ERROR_CHECK(cudaHostAlloc((void **)&h_A, bytes, cudaHostAllocMapped));
    ERROR_CHECK(cudaHostAlloc((void **)&h_B, bytes, cudaHostAllocMapped));

    initializeData<float>(h_A, size);
    initializeData<float>(h_B, size);
    memset(hostRef, 0, bytes);
    memset(gpuRef, 0, bytes);

    // 在使用cudaHostGetDevicePointer之前必须将cudaDeviceMapHost标志传给cudaSetDeviceFlags
    ERROR_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    // 最后一个参数flags是保留参数，当前版本的CUDA必须设置为0
    ERROR_CHECK(cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0));
    ERROR_CHECK(cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0));

    sumArraysHost(h_A, h_B, hostRef, size);
    sumArraysZeroCopy<<<grid, block>>>(d_A, d_B, d_C, size);

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
