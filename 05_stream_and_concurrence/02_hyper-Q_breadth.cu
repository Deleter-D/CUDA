#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

#define SIZE 300000

__global__ void kernel_1()
{
    double sum = 0.0;

    for (int i = 0; i < SIZE; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;

    for (int i = 0; i < SIZE; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;

    for (int i = 0; i < SIZE; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;

    for (int i = 0; i < SIZE; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char const* argv[])
{
    int stream_count = 4;
    int size         = 1 << 12;
    int block_size   = 512;

    if (argc > 1) stream_count = atoi(argv[1]);

    char* env_name = "CUDA_DEVICE_MAX_CONNECTIONS";
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

    cudaStream_t* streams = (cudaStream_t*)malloc(stream_count * sizeof(cudaStream_t));

    for (int i = 0; i < stream_count; i++)
    {
        ERROR_CHECK(cudaStreamCreate(&(streams[i])));
    }

    dim3 block(block_size);
    dim3 grid((size + block.x - 1) / block.x);

    float elapsedTime;
    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    ERROR_CHECK(cudaEventRecord(start));

    // 深度优先调度
    for (int i = 0; i < stream_count; i++)
    {
        kernel_1<<<grid, block, 0, streams[i]>>>();
    }
    for (int i = 0; i < stream_count; i++)
    {
        kernel_2<<<grid, block, 0, streams[i]>>>();
    }
    for (int i = 0; i < stream_count; i++)
    {
        kernel_3<<<grid, block, 0, streams[i]>>>();
    }
    for (int i = 0; i < stream_count; i++)
    {
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }

    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));

    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Measured time for parallel execution = %f ms\n", elapsedTime);

    for (int i = 0; i < stream_count; i++)
    {
        ERROR_CHECK(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    ERROR_CHECK(cudaEventDestroy(start));
    ERROR_CHECK(cudaEventDestroy(stop));

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
