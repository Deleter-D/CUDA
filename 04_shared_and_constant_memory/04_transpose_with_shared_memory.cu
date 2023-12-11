#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析每次请求的全局内存事务数量
    sudo ncu --target-processes all --kernel-name regex:"[a-z]*" --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio /path/to/04_transpose_with_shared_memory

    使用如下命令分析共享内存事务数量
    sudo ncu --target-processes all --kernel-name regex:"[a-z]*" --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum /path/to/04_transpose_with_shared_memory
*/

#define BDIMX 32
#define BDIMY 16
#define PAD 2
#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL)) // 定义行优先索引

// 主机端转置
void transposeHost(float *out, float *in, const int rows, const int cols)
{
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            out[INDEX(col, row, rows)] = in[INDEX(row, col, cols)];
        }
    }
}

// 朴素转置：读为合并访问，写为交叉访问
__global__ void naiveGmem(float *out, float *in, const int rows, const int cols)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
    {
        out[INDEX(col, row, rows)] = in[INDEX(row, col, cols)];
    }
}

// 近似性能上界：读写均为合并访问
__global__ void copyGmem(float *out, float *in, const int rows, const int cols)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
    {
        out[INDEX(row, col, cols)] = in[INDEX(row, col, cols)];
    }
}

// 使用共享内存的转置
__global__ void transposeSmem(float *out, float *in, const int rows, const int cols)
{
    __shared__ float tile[BDIMY][BDIMX];

    // 原始矩阵索引
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
    {
        tile[threadIdx.y][threadIdx.x] = in[INDEX(row, col, cols)];
    }

    // 由于转置过程中，不仅block需要转置，block内的thread也需要转置
    // 所以利用irow和icol来代替原来的threadIdx的x和y维度
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // 转置矩阵中，blockDim和blockIdx的x维度计算列索引，y维度计算行索引，与原始矩阵相反
    row = blockIdx.x * blockDim.x + irow;
    col = blockIdx.y * blockDim.y + icol;

    __syncthreads();

    if (row < cols && col < rows)
    {
        out[INDEX(row, col, rows)] = tile[icol][irow];
    }
}

// 使用填充共享内存的转置
__global__ void transposeSmemPad(float *out, float *in, int rows, int cols)
{
    __shared__ float tile[BDIMY][BDIMX + PAD];

    // 原始矩阵索引
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
    {
        tile[threadIdx.y][threadIdx.x] = in[INDEX(row, col, cols)];
    }

    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    row = blockIdx.x * blockDim.x + irow;
    col = blockIdx.y * blockDim.y + icol;

    __syncthreads();

    if (row < cols && col < rows)
    {
        out[INDEX(row, col, rows)] = tile[icol][irow];
    }
}

// 使用展开的转置
__global__ void transposeSmemUnrollPad(float *out, float *in, int rows, int cols)
{
    // 使用一维的共享内存
    __shared__ float tile[BDIMY][BDIMX * 2 + PAD];

    // 原始矩阵索引
    unsigned int col = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col + blockDim.x < cols)
    {
        tile[threadIdx.y][threadIdx.x] = in[INDEX(row, col, cols)];
        tile[threadIdx.y][threadIdx.x + blockDim.x] = in[INDEX(row, col + blockDim.x, cols)];
    }

    __syncthreads();

    // 转置block中的线程索引
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    row = 2 * blockIdx.x * blockDim.x + irow;
    col = blockIdx.y * blockDim.y + icol;

    if (row + blockDim.x < cols && col < rows)
    {
        out[INDEX(row, col, rows)] = tile[icol][irow];
        out[INDEX(row + blockDim.x, col, rows)] = tile[icol][irow + blockDim.x];
    }
}

int main(int argc, char const *argv[])
{
    setDevice();

    int rows = 1 << 12;
    int cols = 1 << 12;

    printf("Matrix Size: %dx%d\n", rows, cols);
    size_t elem = rows * cols;
    size_t bytes = elem * sizeof(float);

    dim3 block(BDIMX, BDIMY);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    float *h_A = (float *)malloc(bytes);
    float *host_Ref = (float *)malloc(bytes);
    float *gpu_Ref = (float *)malloc(bytes);

    initializeData<float>(h_A, elem);

    transposeHost(host_Ref, h_A, rows, cols);

    float *d_A, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemset(d_C, 0, bytes));
    memset(gpu_Ref, 0, bytes);

    float effectiveBandwidth;
    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    printf("Kernel\t\t\tElapsed time\tEffective bandwidth\n");

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    copyGmem<<<grid, block>>>(d_C, d_A, rows, cols);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    // 近似性能上界
    ERROR_CHECK(cudaEventRecord(start));
    copyGmem<<<grid, block>>>(d_C, d_A, rows, cols);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    effectiveBandwidth = 2 * bytes / 1e9 / elapsedTime;
    printf("copyGmem\t\t%f ms\t%f GB/s\n", elapsedTime, effectiveBandwidth);

    // 朴素转置
    memset(gpu_Ref, 0, bytes);
    ERROR_CHECK(cudaMemset(d_C, 9, bytes));
    ERROR_CHECK(cudaEventRecord(start));
    naiveGmem<<<grid, block>>>(d_C, d_A, rows, cols);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(gpu_Ref, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    effectiveBandwidth = 2 * bytes / 1e9 / elapsedTime;
    printf("naiveGmem\t\t%f ms\t%f GB/s\n", elapsedTime, effectiveBandwidth);
    checkResult<float>(host_Ref, gpu_Ref, elem);

    // 使用共享内存的转置
    memset(gpu_Ref, 0, bytes);
    ERROR_CHECK(cudaMemset(d_C, 9, bytes));
    ERROR_CHECK(cudaEventRecord(start));
    transposeSmem<<<grid, block>>>(d_C, d_A, rows, cols);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(gpu_Ref, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    effectiveBandwidth = 2 * bytes / 1e9 / elapsedTime;
    printf("transposeSmem\t\t%f ms\t%f GB/s\n", elapsedTime, effectiveBandwidth);
    checkResult<float>(host_Ref, gpu_Ref, elem);

    // 使用填充共享内存的转置
    memset(gpu_Ref, 0, bytes);
    ERROR_CHECK(cudaMemset(d_C, 9, bytes));
    ERROR_CHECK(cudaEventRecord(start));
    transposeSmemPad<<<grid, block>>>(d_C, d_A, rows, cols);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(gpu_Ref, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    effectiveBandwidth = 2 * bytes / 1e9 / elapsedTime;
    printf("transposeSmemPad\t%f ms\t%f GB/s\n", elapsedTime, effectiveBandwidth);
    checkResult<float>(host_Ref, gpu_Ref, elem);

    // 使用展开的转置
    memset(gpu_Ref, 0, bytes);
    ERROR_CHECK(cudaMemset(d_C, 9, bytes));
    ERROR_CHECK(cudaEventRecord(start));
    dim3 grid2((grid.x + 2 - 1) / 2, grid.y);
    transposeSmemUnrollPad<<<grid2, block>>>(d_C, d_A, rows, cols);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(gpu_Ref, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    effectiveBandwidth = 2 * bytes / 1e9 / elapsedTime;
    printf("transposeSmemUnrollPad\t%f ms\t%f GB/s\n", elapsedTime, effectiveBandwidth);
    checkResult<float>(host_Ref, gpu_Ref, elem);

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_C));
    free(h_A);
    free(host_Ref);
    free(gpu_Ref);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
