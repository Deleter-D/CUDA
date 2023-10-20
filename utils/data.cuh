#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/**
 * @brief 初始化数据
 * @param ptr 数据指针
 * @param size 数组长度
 */
template <typename T>
void initializaData(T *ptr, int size)
{
    for (int i = 0; i < size; i++)
    {
        ptr[i] = (T)(rand() & 0xFF) / 10.f;
    }
}

/**
    @brief 检查运算结果

    @param hostRef 主机端运算结果
    @param gpuRef  设备端运算结果
    @param N       数组长度
    @param rtol    相对误差参数
    @param atol    绝对误差参数

    @details 如果满足不等式abs(a - b) > (atol + rtol * abs(b))则认为不匹配
*/
template <typename T>
void checkResult(T *hostRef, T *gpuRef, const int N, double rtol = 1e-5, double atol = 1e-8)
{
    std::vector<int> different_indexes;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > (atol + rtol * abs(gpuRef[i])))
        {
            different_indexes.push_back(i);
        }
    }
    int dismatchCount = different_indexes.size();
    if (dismatchCount != 0)
    {
        printf("Arrays do not match!\n\tindexes: ");
        if (dismatchCount < 10)
        {
            for (int i = 0; i < dismatchCount; i++)
                printf("%d ", different_indexes[i]);
            printf("\n");
        }
        else
        {
            for (int i = 0; i < 7; i++)
                printf("%d ", different_indexes[i]);
            printf("... ");
            for (int i = dismatchCount - 3; i < dismatchCount; i++)
                printf("%d ", different_indexes[i]);
            printf("\ttotal: %d\n", dismatchCount);
        }
    }
    else
    {
        printf("Arrays match!\n");
    }
}