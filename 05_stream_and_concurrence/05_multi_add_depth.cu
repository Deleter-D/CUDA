#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

void sumArraysOnHost(float *A, float *B, float *C, const int size)
{
    for (int idx = 0; idx < size; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArrays(float *A, float *B, float *C, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        for (int i = 0; i < 300; ++i) // 增加线程执行时间，便于在nsys中观察
        {
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main(int argc, char const *argv[])
{
    int stream_count = 4;

    if (argc > 1) stream_count = atoi(argv[1]);

    // 通过环境变量调整流的行为
    char *env_name = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(env_name, "32", 1);
    printf("%s = %s\n", env_name, getenv(env_name));

    setDevice();

    cudaDeviceProp prop;
    ERROR_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5))
    {
        if (prop.concurrentKernels == 0)
        {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support Hyper-Q\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n", prop.major, prop.minor, prop.multiProcessorCount);

    int size     = 1 << 18;
    size_t bytes = size * sizeof(float);
    printf("Array size: %d\n", size);

    float *h_A, *h_B, *hostRef, *gpuRef;
    ERROR_CHECK(cudaHostAlloc((void **)&h_A, bytes, cudaHostAllocDefault));
    ERROR_CHECK(cudaHostAlloc((void **)&h_B, bytes, cudaHostAllocDefault));
    ERROR_CHECK(cudaHostAlloc((void **)&hostRef, bytes, cudaHostAllocDefault));
    ERROR_CHECK(cudaHostAlloc((void **)&gpuRef, bytes, cudaHostAllocDefault));

    initializeData<float>(h_A, size);
    initializeData<float>(h_B, size);
    memset(hostRef, 0, bytes);
    memset(gpuRef, 0, bytes);

    sumArraysOnHost(h_A, h_B, hostRef, size);

    float *d_A, *d_B, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_B, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    dim3 block(128);
    dim3 grid((size + block.x - 1) / block.x);

    ERROR_CHECK(cudaEventRecord(start));
    ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    float memcpy_h2d_time;
    ERROR_CHECK(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    float kernel_time;
    ERROR_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    float memcpy_d2h_time;
    ERROR_CHECK(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));

    float total_time = memcpy_h2d_time + kernel_time + memcpy_d2h_time;

    checkResult<float>(hostRef, gpuRef, size);

    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device:\t\t%f ms (%f GB/s)\n", memcpy_h2d_time, (bytes * 1e-6) / memcpy_h2d_time);
    printf(" Kernel execution:\t\t%f ms (%f GB/s)\n", kernel_time, (bytes * 1e-6) / kernel_time);
    printf(" Memcpy device to host:\t\t%f ms (%f GB/s)\n", memcpy_d2h_time, (2 * bytes * 1e-6) / memcpy_d2h_time);
    printf(" Total:\t\t\t\t%f ms (%f GB/s)\n", total_time, (2 * bytes * 1e-6) / total_time);

    // 网格级并行
    int size_per_stream     = size / stream_count;
    size_t bytes_per_stream = size_per_stream * sizeof(float);

    dim3 block2(128);
    dim3 grid2((size_per_stream + block2.x - 1) / block2.x);

    cudaStream_t streams[stream_count];
    for (int i = 0; i < stream_count; i++)
    {
        ERROR_CHECK(cudaStreamCreate(&streams[i]));
    }

    ERROR_CHECK(cudaEventRecord(start));

    for (int i = 0; i < stream_count; i++)
    {
        int offset = i * size_per_stream;
        ERROR_CHECK(cudaMemcpyAsync(&d_A[offset], &h_A[offset], bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));
        ERROR_CHECK(cudaMemcpyAsync(&d_B[offset], &h_B[offset], bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));
        sumArrays<<<grid2, block2, 0, streams[i]>>>(&d_A[offset], &d_B[offset], &d_C[offset], size_per_stream);
        ERROR_CHECK(cudaMemcpyAsync(&gpuRef[offset], &d_C[offset], bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]));
    }

    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    float stream_exec_time;
    ERROR_CHECK(cudaEventElapsedTime(&stream_exec_time, start, stop));

    printf("Result from overlapping data transfers:\n");
    printf(" overlap with %d streams:\t%f ms (%f GB/s)\n", stream_count, stream_exec_time, (2 * bytes * 1e-6) / stream_exec_time);
    printf(" speedup:\t\t\t%f \n", (total_time - stream_exec_time) * 100.0f / total_time);

    ERROR_CHECK(cudaGetLastError());

    checkResult<float>(hostRef, gpuRef, size);

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_B));
    ERROR_CHECK(cudaFree(d_C));

    ERROR_CHECK(cudaFreeHost(h_A));
    ERROR_CHECK(cudaFreeHost(h_B));
    ERROR_CHECK(cudaFreeHost(hostRef));
    ERROR_CHECK(cudaFreeHost(gpuRef));

    ERROR_CHECK(cudaEventDestroy(start));
    ERROR_CHECK(cudaEventDestroy(stop));

    for (int i = 0; i < stream_count; i++)
    {
        ERROR_CHECK(cudaStreamDestroy(streams[i]));
    }

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
