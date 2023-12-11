#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"

/*
  由于NV从计算能力7.0之后就弃用了nvprof，而笔者的设备计算能力为8.9，故这里使用ncu来代替nvprof

  在编译本文件是加上-G参数来防止nvcc借助分支预测优化代码
  nvcc -g -G /path/to/01_warp_divergence.cu -o /path/to/01_warp_divergence -arch sm_89

  使用ncu分析三个核函数的分支效率
  sudo ncu --target-processes all -k regex:"mathKernel*" --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./01_warp_divergence
*/

// 线程ID为偶数的执行if，线程ID为奇数的执行else
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if (tid % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;
}

// 线程束ID为偶数的执行if， 线程束ID为奇数的执行else
__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;
}

// 线程ID为偶数的执行if，线程ID为奇数的执行另一个if
__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    bool ipred = (tid % 2 == 0);
    if (ipred)
    {
        a = 100.0f;
    }
    if (!ipred)
    {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void warmingup(float *c)
{
    // DO NOTHING
}

int main(int argc, char const *argv[])
{
    setDevice();

    int size = 64;
    printf("Data size: %d\n", size);

    dim3 block(64);
    dim3 grid((size + block.x - 1) / block.x);

    float *d_C;
    cudaMalloc((void **)&d_C, size * sizeof(float));

    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));
    float elapsed_time;

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    cudaEventQuery(start);
    mathKernel2<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    mathKernel1<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("mathKernel1<<<%d, %d>>> elapsed %g ms\n", grid.x, block.x, elapsed_time);

    ERROR_CHECK(cudaEventRecord(start));
    mathKernel2<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("mathKernel2<<<%d, %d>>> elapsed %g ms\n", grid.x, block.x, elapsed_time);

    ERROR_CHECK(cudaEventRecord(start));
    mathKernel3<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("mathKernel3<<<%d, %d>>> elapsed %g ms\n", grid.x, block.x, elapsed_time);

    return 0;
}