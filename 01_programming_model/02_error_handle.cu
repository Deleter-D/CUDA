#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
    CUDA运行时的大多数API都支持返回错误代码，返回类型是cudaError_t

    cudaError_t是cudaError枚举类的类型别名，其中含有很多成员，分别代表了不同的错误类型

    CUDA提供了错误检查函数

    
*/
