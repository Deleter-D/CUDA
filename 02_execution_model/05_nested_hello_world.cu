#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"

/*
    使用下列命令编译
    nvcc -g /path/to/05_nestedHelloWorld.cu -o /path/to/05_nestedHelloWorld -arch sm_89 -rdc true

    注意需要编译选项-rdc为true，一些资料中提到还需要链接cudadevrt库，但笔者这里没有显式链接也正常执行了，推测是自动链接了

    若不能正常执行，请使用下列命令编译
    nvcc -g /path/to/05_nestedHelloWorld.cu -o /path/to/05_nestedHelloWorld -arch sm_89 -rdc true -lcudadevrt
*/

__global__ void nestedHelloWorld(int const size, int depth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", depth, tid, threadIdx.x);

    if (size == 1)
        return;

    int threads = size >> 1;
    if (tid == 0 && threads > 0)
    {
        nestedHelloWorld<<<1, threads>>>(threads, ++depth);
        printf("-------> nested execution depth: %d\n", depth);
    }
}

int main(int argc, char const *argv[])
{
    int size = 8;
    int blocksize = 8;
    int gridsize = 1;

    if (argc > 1)
    {
        gridsize = atoi(argv[1]);
        size = gridsize * blocksize;
    }

    setDevice();

    dim3 block(blocksize);
    dim3 grid((size + block.x - 1) / block.x);
    printf("Execution Configuration: grid %d block %d\n", grid.x, block.x);

    nestedHelloWorld<<<grid, block>>>(block.x, 0);

    ERROR_CHECK(cudaDeviceReset());
    return 0;
}
