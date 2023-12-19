#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

#define SIZE (1 << 18)

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data)
{
    printf("callback from stream %d\n", (int*)data);
}

void CUDART_CB my_callback2(void* data)
{
    printf("callback from stream %d\n", (int*)data);
}

__global__ void kernel_1(float* data)
{
    float sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += sum + tan(0.1) * tan(0.1);
    }
    *data = sum;
}

__global__ void kernel_2(float* data)
{
    float sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += sum + tan(0.1) * tan(0.1);
    }
    *data = sum;
}

__global__ void kernel_3(float* data)
{
    float sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += sum + tan(0.1) * tan(0.1);
    }
    *data = sum;
}

__global__ void kernel_4(float* data)
{
    float sum = 0.0;
    for (int i = 0; i < SIZE; i++)
    {
        sum += sum + tan(0.1) * tan(0.1);
    }
    *data = sum;
}

int main(int argc, char const* argv[])
{
    int stream_count = 4;

    if (argc > 1) stream_count = atoi(argv[1]);

    // 通过环境变量调整流的行为
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

    float* d_data;
    ERROR_CHECK(cudaMalloc((void**)&d_data, sizeof(float)));

    cudaStream_t* streams = (cudaStream_t*)malloc(stream_count * sizeof(cudaStream_t));

    for (int i = 0; i < stream_count; i++)
    {
        ERROR_CHECK(cudaStreamCreate(&(streams[i])));
    }

    dim3 block(1);
    dim3 grid(1);

    float elapsedTime;
    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    ERROR_CHECK(cudaEventRecord(start));

    // 深度优先调度
    for (int i = 0; i < stream_count; i++)
    {
        kernel_1<<<grid, block, 0, streams[i]>>>(d_data);
        kernel_2<<<grid, block, 0, streams[i]>>>(d_data);
        kernel_3<<<grid, block, 0, streams[i]>>>(d_data);
        kernel_4<<<grid, block, 0, streams[i]>>>(d_data);

        ERROR_CHECK(cudaStreamAddCallback(streams[i], my_callback, (void*)i, 0));
        ERROR_CHECK(cudaLaunchHostFunc(streams[i], my_callback2, (void*)i));
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
