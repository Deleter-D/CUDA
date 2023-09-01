#include <stdio.h>

// 关于线程索引的数据结构均定义在该头文件中
#include <device_launch_parameters.h>

// 一维grid一维block的情况
__global__ void hello_from_gpu_dim1()
{
    // block索引和thread索引的获取方式
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // 线程唯一ID的获取方式
    const int Idx = blockDim.x * blockIdx.x + threadIdx.x;

    printf("The unique thread id %d is the Thread %d in the Block %d: Hello World from the GPU!\n", Idx, tid, bid);
}

// 二维grid二维block的情况
__global__ void hello_from_gpu_dim2()
{
    // block索引和thread索引的获取方式
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    // 线程唯一ID的获取方式
    const int bId = gridDim.x * blockIdx.y + blockIdx.x;    // block在grid中的唯一ID
    const int tId = blockDim.x * threadIdx.y + threadIdx.x; // thread在block中的唯一ID
    const int Id = bId * blockDim.x * blockDim.y + tId;     // thread在grid中的唯一ID

    printf("The unique thread id %d is the Thread (%d, %d) in the Block (%d, %d): Hello World from the GPU!\n", Id, tidx, tidy, bidx, bidy);
}

// 三维grid三维block的情况
__global__ void hello_from_gpu_dim3()
{
    // block索引和thread索引的获取方式
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tidz = threadIdx.z;
    // 线程唯一ID的获取方式
    const int bId = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;       // block在grid中的唯一ID
    const int tId = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x; // thread在block中的唯一ID
    const int Id = bId * blockDim.x * blockDim.y * blockDim.z + tId;                                // thread在grid中的唯一ID

    printf("The unique thread id %d is the Thread (%d, %d, %d) in the Block (%d, %d, %d): Hello World from the GPU!\n", Id, tidx, tidy, tidz, bidx, bidy, bidz);
}

int main(int argc, char const *argv[])
{
    // 一维的情况
    printf("==========One-dimensional case.==========\n");
    // 每个grid中有2个block，每个block中有4个thread，总共8个thread
    dim3 gridsize_dim1(2);
    dim3 blocksize_dim1(4);
    hello_from_gpu_dim1<<<gridsize_dim1, blocksize_dim1>>>(); // 可以直接写<<<2, 4>>>
    cudaDeviceSynchronize();

    // 二维的情况
    printf("==========Two-dimensional case.==========\n");
    dim3 gridsize_dim2(2, 2);  // 等价于dim3 gridsize(2, 2, 1);
    dim3 blocksize_dim2(2, 2); // 等价于dim3 blocksize(2, 2, 1);
    // 总共2*2*2*2 = 16个thread
    hello_from_gpu_dim2<<<gridsize_dim2, blocksize_dim2>>>();
    cudaDeviceSynchronize();

    // 三维的情况
    printf("==========Three-dimensional case.==========\n");
    dim3 gridsize_dim3(2, 2, 3);
    dim3 blocksize_dim3(2, 2, 2);
    // 总共2*2*3*2*2*2=96个线程
    hello_from_gpu_dim3<<<gridsize_dim3, blocksize_dim3>>>();
    cudaDeviceSynchronize();

    return 0;
}

/*
    根据grid和block的维度不同，可以分为九种情况

    一维grid一维block：
        int bId = blockIdx.x;
        int tId = threadIdx.x;
        int Id = bId * blockDim.x + tId

    一维grid二维block：
        int bId = blockIdx.x;
        int tId = blockDim.x * threadIdx.y + threadIdx.x;
        int Id = bId * blockDim.x * blockDim.y + tId;

    一维grid三维block：
        int bId = blockIdx.x;
        int tId = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
        int Id = bId * blockDim.x * blockDim.y * blockDim.z + tId;

    二维grid一维block：
        int bId = gridDim.x * blockIdx.y + blockIdx.x;
        int tId = threadIdx.x;
        int Id = bId * blockDim.x + tId;

    二维grid二维block：
        int bId = gridDim.x * blockIdx.y + blockIdx.x;
        int tId = blockDim.x * threadIdx.y + threadIdx.x;
        int Id = bId * blockDim.x * blockDim.y + tId;

    二维grid三维block：
        int bId = gridDim.x * blockIdx.y + blockIdx.x;
        int tId = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
        int Id = bId * blockDim.x * blockDim.y * blockDim.z + tId;

    三维grid一维block：
        int bId = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
        int tId = threadIdx.x;
        int Id = bId * blockDim.x + tId;

    三维grid二维block：
        int bId = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
        int tId = blockDim.x * threadIdx.y + threadIdx.x;
        int Id = bId * blockDim.x * blockDim.y + tId;

    三维grid三维block：
        int bId = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
        int tId = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
        int Id = bId * blockDim.x * blockDim.y * blockDim.z + tId;
*/