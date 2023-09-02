#include <stdio.h>

/*
    关于虚拟计算能力，使用nvcc编译器的-arch选项指定
    nvcc path/to/01_compute_capability.cu -o path/to/out/01_compute_capability_89 -arch=compute_89
    笔者的GPU是Geforce RTX 4070Ti，计算能力为8.9，请根据自身GPU情况修改编译命令

    上面的命令编译出了适合本机架构的可执行文件，下面再分别编译出虚拟架构低于本机和高于本机的可执行文件
    nvcc path/to/01_compute_capability.cu -o path/to/out/01_compute_capability_50 -arch=compute_50
    nvcc path/to/01_compute_capability.cu -o path/to/out/01_compute_capability_90 -arch=compute_90

    分别执行三个可执行文件，会发现计算能力89和50的均正常执行，但90的只执行的主机代码，设备代码并未执行
*/

/*
    关于实际计算能力，使用nvcc编译器的-code选项指定
    nvcc path/to/01_compute_capability.cu -o path/to/out/01_compute_capability_89_sm_89 -arch=compute_89 -code=sm_89
    下面再分别编译出实际架构次版本号低于本机本机的可执行文件
    nvcc path/to/01_compute_capability.cu -o path/to/out/01_compute_capability_86_sm_86 -arch=compute_86 -code=sm_86、

    均可以正常执行，由于笔者的GPU计算能力已经是8.x的最高计算能力了，就不再展示次版本号高于本机的情况了
*/

/*
    关于多版本计算能力编译，使用nvcc编译器的-gencode选项
    nvcc path/to/01_compute_capability.cu -o path/to/out/01_compute_capability_fat \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_89,code=sm_89
*/

/*
    关于即时编译
    nvcc path/to/01_compute_capability.cu -o path/to/out/01_compute_capability_jit \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_70,code=\"compute_70,sm_70\"
    这个编译命令并没有编译出针对本机计算能力8.9的二进制代码，但是有针对虚拟架构计算能力7.0的PTX代码
    所以即使本机的计算能力高于7.0,但仍然可以利用PTX采用即时编译的方式正常执行代码
*/

/*
    关于PTX代码
    nvcc path/to/01_compute_capability.cu -ptx
*/

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    printf("Hello World from block %d and thread %d, global id %d\n", bid, tid, id);
}

int main(int argc, char const *argv[])
{
    printf("Hello World from CPU\n");
    hello_from_gpu<<<2, 2>>>();
    cudaDeviceSynchronize();

    return 0;
}