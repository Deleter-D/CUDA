#pragma once
#include <stdio.h>
#include <stdlib.h>

// 错误检查宏函数
#define ERROR_CHECK(call)                                                                           \
    {                                                                                               \
        const cudaError_t error = call;                                                             \
        if (error != cudaSuccess)                                                                   \
        {                                                                                           \
            printf("Error: %s:%d,\n", __FILE__, __LINE__);                                          \
            printf("\tcode: %s, reason: %s\n", cudaGetErrorName(error), cudaGetErrorString(error)); \
        }                                                                                           \
    }

// 错误检查普通函数
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

// 初始化设备
void setDevice()
{
    int deviceCount = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&deviceCount), __FILE__, __LINE__); // 获取设备数量
    // 该API返回一个cudaError_t的枚举类
    if (error != cudaError_t::cudaSuccess || deviceCount == 0)
    {
        printf("Get device count faild. There is no device in your computer.\n");
        exit(-1);
    }
    else
    {
        printf("Get device count successfully.\n");
        printf("There %s %d device%s in your computer.\n", (deviceCount > 1 ? "are" : "is"), deviceCount, (deviceCount > 1 ? "s" : ""));
    }

    int device = 0;
    error = ErrorCheck(cudaSetDevice(device), __FILE__, __LINE__); // 设置执行设备代码的目标设备
    if (error != cudaSuccess)
    {
        printf("Fail to set GPU %d for computing.\n", device);
        exit(-1);
    }
    else
    {
        printf("Set GPU %d for computing.\n", device);
    }
}
