#pragma once
#include <stdio.h>
#include <stdlib.h>

void setDevice()
{
    int deviceCount = 0;
    cudaError_t flag = cudaGetDeviceCount(&deviceCount); // 获取设备数量
    // 该API返回一个cudaError_t的枚举类
    if (flag != cudaError_t::cudaSuccess || deviceCount == 0)
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
    flag = cudaSetDevice(device); // 设置执行设备代码的目标设备
    if (flag != cudaSuccess)
    {
        printf("Fail to set GPU %d for computing.\n", device);
        exit(-1);
    }
    else
    {
        printf("Set GPU %d for computing.\n", device);
    }
}