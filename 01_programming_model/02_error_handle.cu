#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
    CUDA运行时的大多数API都支持返回错误代码，返回类型是cudaError_t

    cudaError_t是cudaError枚举类的类型别名，其中含有很多成员，分别代表了不同的错误类型

    CUDA提供了错误检查函数
        - cudaGetErrorName(cudaError_t e)：返回错误代码对应的名称
        - cudaGetErrorString(cudaError_t e)：返回错误代码描述信息

    常见的错误处理手段是定义一个宏函数或者封装一个普通函数来处理错误
*/

// 定义宏函数的方式
#define ERROR_CHECK(call)                                                                           \
    {                                                                                               \
        const cudaError_t error = call;                                                             \
        if (error != cudaSuccess)                                                                   \
        {                                                                                           \
            printf("Error: %s:%d,\n", __FILE__, __LINE__);                                          \
            printf("\tcode: %s, reason: %s\n", cudaGetErrorName(error), cudaGetErrorString(error)); \
        }                                                                                           \
    }

// 定义普通函数的方式
cudaError_t ErrorCheck(cudaError_t error, const char *filename, int lineNumber)
{
    if (error != cudaSuccess)
    {
        printf("Error: %s:%d,\n", filename, lineNumber);
        printf("\tcode: %s, reason: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        return error;
    }
    return error;
}
// 上述宏函数和普通函数均可以封装进一个头文件中，详见utils/common.cuh

/*
    由于核函数与主机代码是异步执行的，所以上面提到的方法并不能直接捕获核函数相关的错误
    需要利用两个API来捕获核函数内部的错误，在核函数调用之后加上下面两句即可

    ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
*/

__global__ void kernel()
{
    printf("Hello World from the GPU!\n");
}

int main(int argc, char const *argv[])
{
    // 以分配和释放内存为例
    float *h_ptr;
    h_ptr = (float *)malloc(64);
    memset(h_ptr, 0, 64);

    float *d_ptr;

    // 宏函数的方式
    ERROR_CHECK(cudaMalloc((void **)&d_ptr, 64));
    ERROR_CHECK(cudaMemset(d_ptr, 0, 64));
    ERROR_CHECK(cudaMemcpy(d_ptr, h_ptr, 64, cudaMemcpyHostToDevice));
    kernel<<<2, 1025>>>(); // 核函数错误处理，block中最大包含1024个线程，此处一定会报错
    ERROR_CHECK(cudaGetLastError());
    ERROR_CHECK(cudaDeviceSynchronize());
    ERROR_CHECK(cudaFree(d_ptr));

    // 普通函数的方式
    // ErrorCheck(cudaMalloc((void **)&d_ptr, 64), __FILE__, __LINE__);
    // ErrorCheck(cudaMemset(d_ptr, 0, 64), __FILE__, __LINE__);
    // ErrorCheck(cudaMemcpy(d_ptr, h_ptr, 64, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    // kernel<<<2, 1025>>>(); // 核函数错误处理，block中最大包含1024个线程，此处一定会报错
    // ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
    // ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
    // ErrorCheck(cudaFree(d_ptr), __FILE__, __LINE__);

    free(h_ptr);

    return 0;
}
