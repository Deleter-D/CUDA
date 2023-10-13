#pragma once
#include <stdio.h>
#include <stdlib.h>

// 初始化float数据
void initializaFloatData(float *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.f;
    }
}