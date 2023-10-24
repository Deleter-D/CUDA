#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析存储体冲突
    sudo ncu --target-processes all -k regex:"write" --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum /path/to/01_square_shared_memory
*/

#define BDIMX 32
#define BDIMY 32

// 按行写入，按行读取
__global__ void writeRowReadRow(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

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

// 按行写入，按列读取
__global__ void writeRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

void printData(char *msg, int *in, const int size)
{
    printf("%s: ", msg);

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
        printData("write row read row", gpuRef, nx * ny);

    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    writeColReadCol<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    if (print)
        printData("write col read col", gpuRef, nx * ny);

    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    writeRowReadCol<<<grid, block>>>(d_C);
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    if (print)
        printData("write row read col", gpuRef, nx * ny);

    return 0;
}
