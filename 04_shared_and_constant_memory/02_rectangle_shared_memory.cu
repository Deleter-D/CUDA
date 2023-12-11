#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析存储体冲突
    sudo ncu --target-processes all -k regex:"write" --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum /path/to/02_rectangle_shared_memory
*/

#define BDIMX 32
#define BDIMY 16

#define PAD 2

// 按行写入，按行读取
__global__ void writeRowReadRow(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

    // 全局内存的线性索引
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

// 按列写入，按列读取
__global__ void writeColReadCol(int *out)
{
    __shared__ int tile[BDIMX][BDIMY];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

// 按列写入，按行读取
__global__ void writeColReadRow(int *out)
{
    __shared__ int tile[BDIMX][BDIMY];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // 输出的全局内存中的数据元素是经过转置的，所以需要计算转置矩阵中的新坐标
    unsigned int row = idx / blockDim.y;
    unsigned int col = idx % blockDim.y;

    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();

    out[idx] = tile[row][col];
}

// 按行写入，按列读取
__global__ void writeRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int row = idx / blockDim.y;
    unsigned int col = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[col][row];
}

// 按行写入，按列读取，使用动态共享内存
__global__ void writeRowReadColDynamic(int *out)
{
    extern __shared__ int tile[];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int row = idx / blockDim.y;
    unsigned int col = idx % blockDim.y;

    // 将输出矩阵二维索引转换为一维共享内存索引
    unsigned int col_idx = col * blockDim.x + row;

    tile[idx] = idx;

    __syncthreads();

    out[idx] = tile[col_idx];
}

// 按行写入，按列读取，填充静态共享内存
__global__ void writeRowReadColPadding(int *out)
{
    __shared__ int tile[BDIMY][BDIMX + PAD];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int row = idx / blockDim.y;
    unsigned int col = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[col][row];
}

// 按行写入，按列读取，填充动态共享内存
__global__ void writeRowReadColDynPad(int *out)
{
    extern __shared__ int tile[];

    // 全局内存的线性索引
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int row = g_idx / blockDim.y;
    unsigned int col = g_idx % blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x + PAD) + threadIdx.x;
    unsigned int col_idx = col * (blockDim.x + PAD) + row;

    tile[row_idx] = g_idx;

    __syncthreads();

    out[g_idx] = tile[col_idx];
}

void printData(const char *msg, int *in, const int size)
{
    printf("%s", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%5d", in[i]);
        fflush(stdout);
    }

    printf("\n");
    return;
}

int main(int argc, char const *argv[])
{
    bool print = false;
    if (argc > 1)
        print = atoi(argv[1]);

    setDevice();

    cudaSharedMemConfig config;
    ERROR_CHECK(cudaDeviceGetSharedMemConfig(&config));
    printf("Bank Mode: %s\n", (config == cudaSharedMemBankSizeFourByte ? "4-Byte" : "8-Byte"));

    int nx = BDIMX;
    int ny = BDIMY;

    const char *msg;
    size_t bytes = nx * ny * sizeof(int);

    int *gpuRef;
    gpuRef = (int *)malloc(bytes);

    int *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);

    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    writeRowReadRow<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    if (print)
    {
        msg = "write row read row:\t\t";
        printData(msg, gpuRef, nx * ny);
    }

    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    writeColReadCol<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    if (print)
    {
        msg = "write col read col:\t\t";
        printData(msg, gpuRef, nx * ny);
    }

    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    writeColReadRow<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    if (print)
    {
        msg = "write col read row:\t\t";
        printData(msg, gpuRef, nx * ny);
    }

    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    writeRowReadCol<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    if (print)
    {
        msg = "write row read col:\t\t";
        printData(msg, gpuRef, nx * ny);
    }

    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    writeRowReadColDynamic<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    if (print)
    {
        msg = "write row read col Dyn:\t\t";
        printData(msg, gpuRef, nx * ny);
    }

    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    writeRowReadColPadding<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    if (print)
    {
        msg = "write row read col Pad:\t\t";
        printData(msg, gpuRef, nx * ny);
    }

    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    writeRowReadColDynPad<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    if (print)
    {
        msg = "write row read col DynPad:\t";
        printData(msg, gpuRef, nx * ny);
    }

    return 0;
}