#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

#define RADIUS 4
#define BDIM 32

__constant__ float coef[RADIUS + 1];

#define a0 0.00000f
#define a1 0.80000f
#define a2 -0.20000f
#define a3 0.03809f
#define a4 -0.00357f

void stancli1DHost(float* in, float* out, int size)
{
    for (int i = RADIUS; i < size + RADIUS; i++)
    {
        float tmp = 0.0f;

        tmp = a1 * (in[i + 1] - in[i - 1]) +
              a2 * (in[i + 2] - in[i - 2]) +
              a3 * (in[i + 3] - in[i - 3]) +
              a4 * (in[i + 4] - in[i - 4]);
        out[i] = tmp;
    }
}

// 使用常量内存的模板算法
__global__ void stancliConstant(float* in, float* out, int size)
{
    // 包含光环的共享内存
    __shared__ float smem[BDIM + 2 * RADIUS];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;

    while (idx < size + RADIUS)
    {
        // 共享内存的索引，为模板计算作准备
        int sidx = threadIdx.x + RADIUS;

        // 将数据部分写入共享内存
        smem[sidx] = in[idx];

        // 将光环部分度写入共享内存
        if (threadIdx.x < RADIUS)
        {
            smem[sidx - RADIUS] = in[idx - RADIUS];
            smem[sidx + BDIM]   = in[idx + BDIM];
        }

        __syncthreads();

        float tmp = 0.0f;

#pragma unroll
        for (int i = 1; i <= RADIUS; i++)
        {
            tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
        }

        out[idx] = tmp;

        idx += gridDim.x * blockDim.x;
    }
}

// 使用只读缓存的模板算法
__global__ void stancliReadOnly(float* in, float* out, int size, const float* __restrict__ dcoef)
{
    __shared__ float smem[BDIM + 2 * RADIUS];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;

    while (idx < size + RADIUS)
    {
        int sidx = threadIdx.x + RADIUS;

        smem[sidx] = in[idx];

        if (threadIdx.x < RADIUS)
        {
            smem[sidx - RADIUS] = in[idx - RADIUS];
            smem[sidx + BDIM]   = in[idx + BDIM];
        }

        __syncthreads();

        float tmp = 0.0f;

#pragma unroll
        for (int i = 1; i <= RADIUS; i++)
        {
            tmp += dcoef[i] * (smem[sidx + i] - smem[sidx - i]);
        }

        out[idx] = tmp;

        idx += gridDim.x * blockDim.x;
    }
}

int main(int argc, char const* argv[])
{
    setDevice();

    int size     = 1 << 24;
    size_t bytes = (size + 2 * RADIUS) * sizeof(float);
    printf("Array size: %d\n", size);

    float *h_in, *hostRef, *gpuRef;
    h_in    = (float*)malloc(bytes);
    hostRef = (float*)malloc(bytes);
    gpuRef  = (float*)malloc(bytes);

    float *d_in, *d_out;
    ERROR_CHECK(cudaMalloc((void**)&d_in, bytes));
    ERROR_CHECK(cudaMalloc((void**)&d_out, bytes));

    initializeData<float>(h_in, size + 2 * RADIUS);
    stancli1DHost(h_in, hostRef, size);

    cudaDeviceProp prop;
    ERROR_CHECK(cudaGetDeviceProperties(&prop, 0));
    dim3 block(BDIM);
    printf("maxGridSize: %d\n", prop.maxGridSize[0]);
    dim3 grid(prop.maxGridSize[0] < size / block.x ? prop.maxGridSize[0] : size / block.x);

    float elapsedTime;
    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    const float h_coef[] = {a0, a1, a2, a3, a4};
    // 使用常量内存只需要调用如下运行时API，无需申请设备内存
    ERROR_CHECK(cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float)));

    ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    stancliConstant<<<grid, block>>>(d_in, d_out, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("constant <<<%d, %d>>> elapsed time: %f ms\n", grid.x, block.x, elapsedTime);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_out, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    checkResult<float>(hostRef + RADIUS, gpuRef + RADIUS, size, 1, 1e-5, 1e-5);

    // 使用只读缓存时需要提前申请设备内存
    float* d_coef;
    ERROR_CHECK(cudaMalloc((void**)&d_coef, (RADIUS + 1) * sizeof(float)));
    ERROR_CHECK(cudaMemcpy(d_coef, h_coef, (RADIUS + 1) * sizeof(float), cudaMemcpyHostToDevice));

    memset(gpuRef, 0, bytes);
    ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemset(d_out, 0, bytes));
    ERROR_CHECK(cudaEventRecord(start));
    stancliReadOnly<<<grid, block>>>(d_in, d_out, size, d_coef);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("readOnly <<<%d, %d>>> elapsed time: %f ms\n", grid.x, block.x, elapsedTime);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_out, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    checkResult<float>(hostRef + RADIUS, gpuRef + RADIUS, size, 1, 1e-5, 1e-5);

    ERROR_CHECK(cudaFree(d_in));
    ERROR_CHECK(cudaFree(d_out));
    free(h_in);
    free(hostRef);
    free(gpuRef);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
