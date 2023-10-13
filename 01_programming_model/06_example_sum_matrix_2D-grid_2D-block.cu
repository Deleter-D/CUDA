#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

// Host端计算函数，用来验证核函数结果的正确性
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

// 二维grid二维block的矩阵加法核函数
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x; // 线程在x维度上的索引
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y; // 线程在y维度上的索引
    unsigned int idx = iy * nx + ix;                         // 线程的全局索引

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char const *argv[])
{
    setDevice(); // 之前封装好的设备初始化操作

    // 设置矩阵尺寸
    int nx = 1 << 14; // 16384
    int ny = 1 << 14; // 16384

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: (%d, %d)\n", nx, ny);

    // 申请主机内存
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // 初始化主机端数据
    initializaFloatData(h_A, nxy);
    initializaFloatData(h_B, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // 调用主机端的矩阵加法以备检查核函数运算结果
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    // 申请设备内存
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // 核函数调用
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    printf("sumMatrixOnGPU2D <<<(%d, %d), (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

    // 将运算结果从设备端拷贝到主机端
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // 检查结果正确性
    checkResult(hostRef, gpuRef, nxy);

    // 释放设备内存
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // 重置设备
    cudaDeviceReset();

    return 0;
}
