#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

__device__ int myAtomicAdd(int *address, int incr)
{
    int expected = *address;                                      // 记录当前内存地址的值
    int oldValue = atomicCAS(address, expected, expected + incr); // 尝试增加incr，CAS会返回目标地址的值

    while (oldValue != expected) // 如果返回值与预期值不等，则CAS没有成功
    {
        // 重复执行CAS直到成功
        expected = oldValue;                                      // 获取目标地址的新值
        oldValue = atomicCAS(address, expected, expected + incr); // 继续尝试增加incr
    }

    return oldValue; // 为了匹配其他CUDA原子函数的语义，这里返回目标地址的值
}

__global__ void kernel(int *sharedInteger)
{
    myAtomicAdd(sharedInteger, 1);
}

int main(int argc, char const *argv[])
{
    int *d_sharedInteger, h_sharedInteger;
    ERROR_CHECK(cudaMalloc((void **)&d_sharedInteger, sizeof(int)));
    ERROR_CHECK(cudaMemset(d_sharedInteger, 0, sizeof(int)));

    kernel<<<4, 128>>>(d_sharedInteger);

    ERROR_CHECK(cudaMemcpy(&h_sharedInteger, d_sharedInteger, sizeof(int), cudaMemcpyDeviceToHost));
    printf("4 x 128 increments led to value of %d\n", h_sharedInteger);

    return 0;
}
