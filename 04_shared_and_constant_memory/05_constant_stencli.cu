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

void setupCoefConstant()
{
    const float h_coef[] = {a0, a1, a2, a3, a4};
    ERROR_CHECK(cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float)));
}

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

__global__ void stancli1DGPU(float* in, float* out, int size)
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

    ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    setupCoefConstant();

    cudaDeviceProp prop;
    ERROR_CHECK(cudaGetDeviceProperties(&prop, 0));
    dim3 block(BDIM);
    printf("maxGridSize: %d\n", prop.maxGridSize[0]);
    dim3 grid(prop.maxGridSize[0] < size / block.x ? prop.maxGridSize[0] : size / block.x);

    printf("<<<%d, %d>>>\n", grid.x, block.x);

    stancli1DGPU<<<grid, block>>>(d_in, d_out, size);

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
