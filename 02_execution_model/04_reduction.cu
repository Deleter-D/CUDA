#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用ncu分析每个线程束上执行的指令数量
    sudo ncu --target-processes all --kernel-name regex:"reduce*" --metrics smsp__average_inst_executed_per_warp.ratio /path/to/out/04_reduction

    使用ncu分析内存读取效率
    sudo ncu --target-processes all --kernel-name regex:"reduce*" --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second /path/to/out/04_reduction

    使用ncu分析内存加载和存储效率
    sudo ncu --target-processes all --kernel-name regex:"reduce*" --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct /path/to/out/04_reduction
*/

// C语言递归实现的归约求和
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

// 相邻匹配的并行归约求和
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int size)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 将全局数据指针转换为当前block的局部数据指针
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // 边界检查
    if (idx >= size)
        return;

    // 在全局内存中原地归约
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads(); // 同步线程，保证下一轮迭代正确
    }

    // 将当前block的结果写入全局内存
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// 强制相邻线程工作的相邻匹配并行归约求和
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= size)
        return;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // 将tid转换为局部数组索引
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// 交错匹配的并行归约求和
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= size)
        return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// 循环展开的交错匹配并行归约求和，汇聚2个数据块
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    // 与之前不同，这里将2个数据块汇总到一个线程块中
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    if (idx + blockDim.x < size)
        g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// 循环展开的交错匹配并行归约求和，汇聚4个数据块
__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + tid;

    // 这里将4个数据块汇总到一个线程块中
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    if (idx + blockDim.x * 3 < size)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// 循环展开的交错匹配并行归约求和，汇聚8个数据块
__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + tid;

    // 这里将8个数据块汇总到一个线程块中
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < size)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        int a5 = g_idata[idx + blockDim.x * 4];
        int a6 = g_idata[idx + blockDim.x * 5];
        int a7 = g_idata[idx + blockDim.x * 6];
        int a8 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// 循环展开的交错匹配并行归约求和，汇聚8个数据块，并展开最后线程束
__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + tid;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < size)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        int a5 = g_idata[idx + blockDim.x * 4];
        int a6 = g_idata[idx + blockDim.x * 5];
        int a7 = g_idata[idx + blockDim.x * 6];
        int a8 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // 最后线程束展开
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

// 完全展开的归约
__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int size)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + tid;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < size)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        int a5 = g_idata[idx + blockDim.x * 4];
        int a6 = g_idata[idx + blockDim.x * 5];
        int a7 = g_idata[idx + blockDim.x * 6];
        int a8 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

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

// 带模板参数的完全展开的归约
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int size)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + tid;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < size)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        int a5 = g_idata[idx + blockDim.x * 4];
        int a6 = g_idata[idx + blockDim.x * 5];
        int a7 = g_idata[idx + blockDim.x * 6];
        int a8 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    if (iBlockSize >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)
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

int main(int argc, char const *argv[])
{
    int blocksize = 512;

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
    warmupKernelDo();
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
    printf("cpu reduce\telapsed %g ms\tcpu_sum: %d\n", elapsed_time_cpu, cpu_sum);

    // 相邻配对的并行归约求和
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu neighbored\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

    // 强制相邻线程工作的相邻匹配并行归约求和
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu neighboredL\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

    // 交错匹配并行归约求和
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu interleaved\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

    // 循环展开的交错匹配并行归约求和，汇聚2个数据块
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size); // 注意grid数量变成了原来的一半
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i++)
        gpu_sum += h_odata[i];
    printf("gpu unrolling2\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x / 2, block.x);

    // 循环展开的交错匹配并行归约求和，汇聚4个数据块
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size); // 注意grid数量变成了原来的1/4
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += h_odata[i];
    printf("gpu unrolling4\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x / 4, block.x);

    // 循环展开的交错匹配并行归约求和，汇聚8个数据块
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size); // 注意grid数量变成了原来的1/8
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];
    printf("gpu unrolling8\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x / 8, block.x);

    // 循环展开的交错匹配并行归约求和，汇聚8个数据块，并展开最后线程束
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];
    printf("gpu unrolWarps8\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x / 8, block.x);

    // 完全展开的归约
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];
    printf("gpu CmptUnroll8\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x / 8, block.x);

    // 带模板参数的完全展开的归约
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    switch (blocksize)
    {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    case 512:
        reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    case 256:
        reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    case 128:
        reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    case 64:
        reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    }
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];
    printf("gpu CmptUnroll\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x / 8, block.x);

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
