#include <stdio.h>

// 关于线程索引的数据结构均定义在该头文件中
#include <device_launch_parameters.h>

__global__ void hello_from_gpu()
{
    printf("Thread %d in Block %d: Hello World from the GPU!\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char const *argv[])
{
    hello_from_gpu<<<2, 4>>>();
    /*
        暂时只考虑一维的情况
        <<<2, 4>>>表示每个grid中有2个block，每个block中有4个thread，总共8个thread

        则相应的索引数据结构对应的值分别为：

        gridDim.x == 2
        blockDim.x == 4

        blockIdx.x的范围是[0, gridDim.x-1]，即[0, 1]
        threadIdx.x的范围是[0, blockDim.x-1]，即[0, 3]
    */

    cudaDeviceSynchronize();
    return 0;
}