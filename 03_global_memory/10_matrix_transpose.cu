#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析全局加载、存储吞吐量和全局加载、存储效率
    sudo ncu --target-processes all -k regex:"transpose" \
        --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
        /path/to/10_matrix_transpose

*/

#define BDIMX 16
#define BDIMY 16

// 主机实现的错位转置算法
void transposeHost(float *out, float *in, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
        for (int ix = 0; ix < nx; ix++)
            out[ix * ny + iy] = in[iy * nx + ix];
}

// 按行读取和存储，计算核函数带宽上限
__global__ void copyRow(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny)
        out[iy * nx + ix] = in[iy * nx + ix];
}

// 按列读取和存储，计算核函数带宽下限
__global__ void copyCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny)
        out[ix * ny + iy] = in[ix * ny + iy];
}

// 按行读取，按列存储的转置
__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny)
        out[ix * ny + iy] = in[iy * nx + ix];
}

// 按列读取，按行存储的转置
__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny)
        out[iy * nx + ix] = in[ix * ny + iy];
}

// 展开的按行读取，按列存储的转置
__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;
    unsigned int to = ix * ny + iy;

    if (ix + blockDim.x * 3 < nx && iy < ny)
    {
        out[to] = in[ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + ny * blockDim.x * 2] = in[ti + blockDim.x * 2];
        out[to + ny * blockDim.x * 3] = in[ti + blockDim.x * 3];
    }
}

// 展开的按列读取，按行存储的转置
__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int ti = ix * ny + iy;
    unsigned int to = iy * nx + ix;
    if (ix + blockDim.x * 3 < nx && iy < ny)
    {
        out[to] = in[ti];
        out[to + blockDim.x] = in[ti + ny * blockDim.x];
        out[to + blockDim.x * 2] = in[ti + ny * blockDim.x * 2];
        out[to + blockDim.x * 3] = in[ti + ny * blockDim.x * 3];
    }
}

int main(int argc, char const *argv[])
{
    setDevice();

    int nx = 1 << 11;
    int ny = 1 << 11;

    int kernelNum = 0;
    int blockx = 16;
    int blocky = 16;
    if (argc > 1)
        kernelNum = atoi(argv[1]);
    if (argc > 2)
        blockx = atoi(argv[2]);
    if (argc > 3)
        blocky = atoi(argv[3]);

    printf("Matrix size: %d x %d, launch kernel %d\n", nx, ny, kernelNum);

    size_t bytes = nx * ny * sizeof(float);

    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    float *h_A, *hostRef, *gpuRef;
    h_A = (float *)malloc(bytes);
    hostRef = (float *)malloc(bytes);
    gpuRef = (float *)malloc(bytes);

    initializeData<float>(h_A, nx * ny);

    transposeHost(hostRef, h_A, nx, ny);

    float *d_A, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    warmupKernelDo();
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    // 核函数指针与描述
    void (*kernel)(float *, float *, int, int);
    char *kernelName;

    switch (kernelNum)
    {
    case 0:
        kernel = &copyRow;
        kernelName = "CopyRow";
        break;
    case 1:
        kernel = &copyCol;
        kernelName = "CopyCol";
        break;
    case 2:
        kernel = &transposeNaiveRow;
        kernelName = "NaiveRow";
        break;
    case 3:
        kernel = &transposeNaiveCol;
        kernelName = "NaiveCol";
        break;
    case 4:
        kernel = &transposeUnroll4Row;
        kernelName = "Unroll4Row";
        break;
    case 5:
        kernel = &transposeUnroll4Col;
        kernelName = "Unroll4Col";
        break;
    }

    // 执行核函数
    ERROR_CHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(d_C, d_A, nx, ny);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    // 获取理论带宽
    cudaDeviceProp prop;
    ERROR_CHECK(cudaGetDeviceProperties(&prop, 0));

    // 计算有效带宽
    float bandWidth = (2 * nx * ny * sizeof(float) * 1.0E-9) / (elapsedTime * 1.0E-3);
    printf("%s\t<<<(%d, %d), (%d, %d)>>>\telapsed %f ms\teffective bandwidth %f GB/s\n", kernelName, grid.x, grid.y, block.x, block.y, elapsedTime, bandWidth);

    if (kernelNum > 1)
    {
        ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
        checkResult<float>(hostRef, gpuRef, nx * ny);
    }

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
