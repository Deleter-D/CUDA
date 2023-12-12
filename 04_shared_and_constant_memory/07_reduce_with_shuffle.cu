#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

#define DIM 128
#define SMEMDIM 4

int recursiveReduce(int *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}

// 使用洗牌指令的并行归约
__inline__ __device__ int warpReduce(int localSum)
{
    localSum += __shfl_xor_sync(0xffff, localSum, 16);
    localSum += __shfl_xor_sync(0xffff, localSum, 8);
    localSum += __shfl_xor_sync(0xffff, localSum, 4);
    localSum += __shfl_xor_sync(0xffff, localSum, 2);
    localSum += __shfl_xor_sync(0xffff, localSum, 1);

    return localSum;
}
__global__ void shflReduce(int *out, int *in, unsigned int size)
{
    __shared__ int smem[SMEMDIM];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    // 计算束内线程索引和线程束索引
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;

    // 逐block进行线程束归约
    int localSum = warpReduce(in[idx]);

    // 将线程束和存入共享内存
    if (laneIdx == 0)
        smem[warpIdx] = localSum;

    __syncthreads();

    // 最后一个线程束归约
    if (threadIdx.x < warpSize)
        localSum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;

    if (warpIdx == 0)
        localSum = warpReduce(localSum);

    // 将结果写回全局内存
    if (threadIdx.x == 0)
        out[blockIdx.x] = localSum;
}

int main(int argc, char const *argv[])
{
    setDevice();

    int size     = 1 << 10;
    size_t bytes = size * sizeof(int);

    dim3 block(DIM);
    dim3 grid((size + block.x - 1) / block.x);

    int cpu_sum = 0, gpu_sum = 0;
    int *h_in, *h_out, *tmp;
    h_in  = (int *)malloc(bytes);
    h_out = (int *)malloc(grid.x * sizeof(int));
    tmp   = (int *)malloc(bytes);

    initializeData<int>(h_in, size);

    memcpy(tmp, h_in, bytes);
    cpu_sum = recursiveReduce(tmp, size);
    printf("cpu sum:\t%d\n", cpu_sum);

    int *d_in, *d_out;
    ERROR_CHECK(cudaMalloc((void **)&d_in, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_out, grid.x * sizeof(int)));

    ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    shflReduce<<<grid, block>>>(d_out, d_in, size);
    ERROR_CHECK(cudaMemcpy(h_out, d_out, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) gpu_sum += h_out[i];
    printf("gpu sum:\t%d\n", gpu_sum);

    ERROR_CHECK(cudaFree(d_in));
    ERROR_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(tmp);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
