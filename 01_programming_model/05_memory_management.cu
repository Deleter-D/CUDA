#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    // >>> 主机内存分配及其初始化 <<<
    int elemCount     = 512;                       // 元素个数
    size_t bytesCount = elemCount * sizeof(float); // 总字节数
    float *h_A, *h_B, *h_C;
    // 申请主机内存
    h_A = (float *)malloc(bytesCount);
    h_B = (float *)malloc(bytesCount);
    h_C = (float *)malloc(bytesCount);
    // 初始化
    memset(h_A, 0, bytesCount);
    memset(h_B, 0, bytesCount);
    memset(h_C, 0, bytesCount);

    // >>> 设备内存分配及其初始化 <<<
    float *d_A, *d_B, *d_C;
    // 申请设备内存
    cudaMalloc((void **)&d_A, bytesCount);
    cudaMalloc((void **)&d_B, bytesCount);
    cudaMalloc((void **)&d_C, bytesCount);
    // 初始化
    cudaMemset(d_A, 0, bytesCount);
    cudaMemset(d_B, 0, bytesCount);
    cudaMemset(d_C, 0, bytesCount);

    // >>> 主机到设备的数据般移（将待计算数据般移到设备）<<<
    cudaMemcpy(d_A, h_A, bytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesCount, cudaMemcpyHostToDevice);

    // 调用核函数...

    // >>> 设备到主机的数据般移（将计算结果般移到主机）<<<
    cudaMemcpy(h_C, d_C, bytesCount, cudaMemcpyDeviceToHost);

    // >>> 设备内存释放 <<<
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // >>> 主机内存释放 <<<
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}