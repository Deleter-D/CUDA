#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析全局内存加载和存储事务
    sudo ncu --target-processes all --kernel-name regex:"reduce*" --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum /path/to/03_reduce_with_shared_memory
*/

#define DIM 128

// 之前实现的线程束展开的并行归约，作为性能基准
__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    // 完全展开
    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();

    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// 线程束展开的并行归约，但使用共享内存
__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int size)
{
    __shared__ int smem[DIM];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    // 将输入写入共享内存
    smem[tid] = idata[tid];
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    if (tid < 32)
    {
        volatile int *vmem = smem;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

// 在上面核函数的基础上加上循环展开
__global__ void reduceSmemUnroll(int *g_idata, int *g_odata, unsigned int size)
{
    __shared__ int smem[DIM];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    int temp_sum = 0;

    if (idx < size)
    {
        int a1, a2, a3, a4;
        a1 = a2 = a3 = a4 = 0;
        a1 = g_idata[idx];
        if (idx + blockDim.x < size)
            a2 = g_idata[idx + blockDim.x];
        if (idx + blockDim.x * 2 < size)
            a3 = g_idata[idx + blockDim.x * 2];
        if (idx + blockDim.x * 3 < size)
            a4 = g_idata[idx + blockDim.x * 3];
        temp_sum = a1 + a2 + a3 + a4;
    }

    // 将输入写入共享内存
    smem[tid] = temp_sum;
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    if (tid < 32)
    {
        volatile int *vmem = smem;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

// 在上面核函数改写为使用动态共享内存
__global__ void reduceSmemUnrollDyn(int *g_idata, int *g_odata, unsigned int size)
{
    extern __shared__ int smem[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    int temp_sum = 0;

    if (idx < size)
    {
        int a1, a2, a3, a4;
        a1 = a2 = a3 = a4 = 0;
        a1 = g_idata[idx];
        if (idx + blockDim.x < size)
            a2 = g_idata[idx + blockDim.x];
        if (idx + blockDim.x * 2 < size)
            a3 = g_idata[idx + blockDim.x * 2];
        if (idx + blockDim.x * 3 < size)
            a4 = g_idata[idx + blockDim.x * 3];
        temp_sum = a1 + a2 + a3 + a4;
    }

    // 将输入写入共享内存
    smem[tid] = temp_sum;
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    if (tid < 32)
    {
        volatile int *vmem = smem;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

int recursiveReduce(int *data, int const size)
{
    if (size == 1)
        return data[0];
    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

int main(int argc, char const *argv[])
{
    int blocksize = DIM;

    if (argc > 1)
        blocksize = atoi(argv[1]);

    setDevice();

    int size = 1 << 24;
    printf("Array Size: %d\n", size);

    dim3 block(blocksize);
    dim3 grid((size + block.x - 1) / block.x);

    int cpu_sum, gpu_sum;

    int *h_idata, *h_odata, *temp;
    h_idata = (int *)malloc(size * sizeof(int));
    h_odata = (int *)malloc(grid.x * sizeof(int));
    temp = (int *)malloc(size * sizeof(int)); // 用于CPU端求和

    initializeData<int>(h_idata, size);
    memcpy(temp, h_idata, size * sizeof(int));

    int *d_idata, *d_odata;
    ERROR_CHECK(cudaMalloc((void **)&d_idata, size * sizeof(int)));
    ERROR_CHECK(cudaMalloc((void **)&d_odata, grid.x * sizeof(int)));

    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    reduceGmem<<<grid.x, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    // 主机端递归实现的归约求和
    struct timeval tp;
    gettimeofday(&tp, NULL);
    double start_cpu = (double)(tp.tv_sec * 1.e3) + (double)(tp.tv_usec * 1.e-3);
    cpu_sum = recursiveReduce(temp, size);
    gettimeofday(&tp, NULL);
    double elapsed_time_cpu = (double)(tp.tv_sec * 1.e3) + (double)(tp.tv_usec * 1.e-3) - start_cpu;
    printf("cpu reduce\t\telapsed %g ms\tcpu_sum: %d\n", elapsed_time_cpu, cpu_sum);

    // 线程束展开的归约
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceGmem<<<grid.x, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu Gemm\t\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

    // 线程束展开的归约，使用共享内存
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceSmem<<<grid.x, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu Semm\t\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

    // 完全展开的归约，使用共享内存
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceSmemUnroll<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_odata[i];
    printf("gpu SemmUnroll\t\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x / 4, block.x);

    // 完全展开的归约，使用动态共享内存
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceSmemUnrollDyn<<<grid.x / 4, block, DIM * sizeof(int)>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_odata[i];
    printf("gpu SemmUnrollDyn\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x / 4, block.x);

    ERROR_CHECK(cudaFree(d_idata));
    ERROR_CHECK(cudaFree(d_odata));

    free(h_idata);
    free(h_odata);
    free(temp);

    ERROR_CHECK(cudaDeviceReset());

    if (cpu_sum == gpu_sum)
        printf("Result correct!\n");
    else
        printf("Result error!\n");

    return 0;
}