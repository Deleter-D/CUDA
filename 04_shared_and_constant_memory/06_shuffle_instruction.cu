#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

#define BDIMX 16
#define SEGM 4

// 广播
__global__ void shflBroadcast(int *out, int *in, int const srcLane)
{
    int value        = in[threadIdx.x];
    value            = __shfl_sync(0xffff, value, srcLane, BDIMX);
    out[threadIdx.x] = value;
}

// 上移
__global__ void shflUp(int *out, int *in, unsigned int const delta)
{
    int value        = in[threadIdx.x];
    value            = __shfl_up_sync(0xffff, value, delta, BDIMX);
    out[threadIdx.x] = value;
}

// 下移
__global__ void shflDown(int *out, int *in, unsigned int const delta)
{
    int value        = in[threadIdx.x];
    value            = __shfl_down_sync(0xffff, value, delta, BDIMX);
    out[threadIdx.x] = value;
}

// 环绕移动
__global__ void shflWarp(int *out, int *in, int const offset)
{
    int value        = in[threadIdx.x];
    value            = __shfl_sync(0xffff, value, threadIdx.x + offset, BDIMX);
    out[threadIdx.x] = value;
}

// 蝴蝶交换
__global__ void shflXor(int *out, int *in, int const mask)
{
    int value        = in[threadIdx.x];
    value            = __shfl_xor_sync(0xffff, value, mask, BDIMX);
    out[threadIdx.x] = value;
}

// 交换数组
__global__ void shflXorArray(int *out, int *in, int const mask)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];
    for (int i = 0; i < SEGM; i++)
    {
        value[i] = in[idx + i];
    }

    value[0] = __shfl_xor_sync(0xf, value[0], mask, BDIMX);
    value[1] = __shfl_xor_sync(0xf, value[1], mask, BDIMX);
    value[2] = __shfl_xor_sync(0xf, value[2], mask, BDIMX);
    value[3] = __shfl_xor_sync(0xf, value[3], mask, BDIMX);

    for (int i = 0; i < SEGM; i++)
    {
        out[idx + i] = value[i];
    }
}

// 使用数组索引交换数值
__inline__ __device__ void swap(int *value, int laneIdx, int mask, int firstIdx, int secondIdx)
{
    // 识别待交换的每对线程中的第一个线程
    bool pred = (laneIdx % 2 == 0);
    if (pred) // 第一个线程的数组firstIdx, secondIdx两个位置交换
    {
        int tmp          = value[firstIdx];
        value[firstIdx]  = value[secondIdx];
        value[secondIdx] = tmp;
    }

    // 两个线程互换数组的secondIdx位置元素
    value[secondIdx] = __shfl_xor_sync(0xf, value[secondIdx], mask, BDIMX);

    if (pred) // 第一个线程的数组firstIdx, secondIdx两个位置再换回来
    {
        int tmp          = value[firstIdx];
        value[firstIdx]  = value[secondIdx];
        value[secondIdx] = tmp;
    }
}
__global__ void shflSwap(int *out, int *in, int const mask, int firstIdx, int secondIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++)
    {
        value[i] = in[idx + i];
    }

    swap(value, threadIdx.x, mask, firstIdx, secondIdx);

    for (int i = 0; i < SEGM; i++)
    {
        out[idx + i] = value[i];
    }
}

void printData(int *in, const int size)
{
    for (int i = 0; i < size; i++)
        printf("%2d ", in[i]);
    printf("\n");
}

int main(int argc, char const *argv[])
{
    setDevice();

    int size     = BDIMX;
    size_t bytes = size * sizeof(int);

    int *h_in, *h_out;
    h_in  = (int *)malloc(bytes);
    h_out = (int *)malloc(bytes);

    for (int i = 0; i < size; i++)
    {
        h_in[i] = i;
    }
    printf("initialData:\t\t");
    printData(h_in, size);

    int *d_in, *d_out;
    ERROR_CHECK(cudaMalloc((void **)&d_in, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_out, bytes));

    ERROR_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(BDIMX);
    dim3 grid(1);

    shflBroadcast<<<grid, block>>>(d_out, d_in, 2);
    ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("After shfl:\t\t");
    printData(h_out, size);

    shflUp<<<grid, block>>>(d_out, d_in, 2);
    ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("After shfl_up:\t\t");
    printData(h_out, size);

    shflDown<<<grid, block>>>(d_out, d_in, 2);
    ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("After shfl_down:\t");
    printData(h_out, size);

    shflWarp<<<grid, block>>>(d_out, d_in, 2);
    ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("After shfl_warp:\t");
    printData(h_out, size);

    shflXor<<<grid, block>>>(d_out, d_in, 1);
    ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("After shfl_xor:\t\t");
    printData(h_out, size);

    shflXorArray<<<grid, block.x / SEGM>>>(d_out, d_in, 1); // 每个线程有4个元素，故线程块缩小到原来的1/4
    ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("After shfl_xor_array:\t");
    printData(h_out, size);

    shflSwap<<<grid, block.x / SEGM>>>(d_out, d_in, 1, 0, 3); // 每个线程有4个元素，故线程块缩小到原来的1/4
    ERROR_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("After shfl_swap:\t");
    printData(h_out, size);

    ERROR_CHECK(cudaFree(d_in));
    ERROR_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
