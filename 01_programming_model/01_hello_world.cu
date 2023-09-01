#include <stdio.h>

__global__ void hello_from_gpu()
{
    // 设备代码不支持C++的<iostream>
    // 输出信息必须使用C的<stdio.h>
    printf("Hello World from the GPU!\n");
}

int main(int argc, char const *argv[])
{
    hello_from_gpu<<<1, 1>>>();
    // 下面的调用将同步主机与设备，促使缓冲区刷新，从而使GPU输出信息到终端
    cudaDeviceSynchronize();
    return 0;
}