#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

void sumArraysHost(float *A, float *B, float *C, const int size, int offset)
{
    for (int i = 0, j = offset; i < size; i++, j++)
        C[i] = A[j] + B[j];
}

__global__ void sumArraysReadOffset(float *A, float *B, float *C, const int size, int offset)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = tid + offset;
    if (tid < size)
        C[tid] = A[j] + B[j];
}

int main(int argc, char const *argv[])
{
    setDevice();

    int size = 1 << 20;
    printf("Array size: %d\n", size);
    size_t bytes = size * sizeof(float);

    int offset = 0;
    if (argc > 1)
        offset = atoi(argv[1]);

    dim3 block(512);
    dim3 grid((size + block.x - 1) / block.x);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(2 * bytes);
    h_B = (float *)malloc(2 * bytes);
    hostRef = (float *)malloc(bytes);
    gpuRef = (float *)malloc(bytes);

    initializaData<float>(h_A, 2 * size);
    initializaData<float>(h_B, 2 * size);

    sumArraysHost(h_A, h_B, hostRef, size, offset);

    float *d_A, *d_B, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, 2 * bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_B, 2 * bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    ERROR_CHECK(cudaMemcpy(d_A, h_A, 2 * bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_B, h_B, 2 * bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    warmupKernelDo();
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    sumArraysReadOffset<<<grid, block>>>(d_A, d_B, d_C, size, offset);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("readOffset<<<%d, %d>>>\toffset %4d\telapsed %f ms\n", grid.x, block.x, offset, elapsedTime);

    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());

    checkResult<float>(hostRef, gpuRef, size);

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_B));
    ERROR_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
