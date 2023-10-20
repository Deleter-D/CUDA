#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

// C语言递归实现的归约求和
int recursiveReduce(int *data, int const size)
{
    if (size == 1)
        return data[0];
    int const stride = size >> 1;

    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

// 嵌套归约
__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    // 递归中止条件
    if (size == 2 && tid == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int stride = size >> 1;
    if (stride > 1 && tid < stride)
    {
        idata[tid] += idata[tid + stride];
    }
    __syncthreads();

    // 嵌套调用生成子网格
    if (tid == 0)
    {
        gpuRecursiveReduce<<<1, stride, 0, cudaStreamTailLaunch>>>(idata, odata, stride);
    }
    __syncthreads();
}

// 去除同步的嵌套归约
__global__ void gpuRecursiveReduceNosync(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    if (size == 2 && tid == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int stride = size >> 1;
    if (stride > 1 && tid < stride)
    {
        idata[tid] += idata[tid + stride];
        if (tid == 0)
        {
            gpuRecursiveReduceNosync<<<1, stride, 0, cudaStreamTailLaunch>>>(idata, odata, stride);
        }
    }
}

// 优化资源利用后的嵌套归约
__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, int stride, int const dim)
{
    int *idata = g_idata + blockIdx.x * dim;

    if (stride == 1 && threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    idata[threadIdx.x] += idata[threadIdx.x + stride];

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        gpuRecursiveReduce2<<<gridDim.x, stride / 2, 0, cudaStreamTailLaunch>>>(g_idata, g_odata, stride / 2, dim);
    }
}

// 相邻匹配的并行归约求和
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int size)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= size)
        return;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char const *argv[])
{
    int nblock = 1024;
    int nthread = 512;

    if (argc > 1)
        nblock = atoi(argv[1]);
    if (argc > 2)
        nthread = atoi(argv[2]);

    setDevice();

    int size = nblock * nthread;
    printf("Array size: %d\n", size);

    dim3 block(nthread);
    dim3 grid((size + block.x - 1) / block.x);
    printf("Execution Configuration: grid %d block %d\n", grid.x, block.x);

    int cpu_sum, gpu_sum;

    int *h_idata, *h_odata, *temp;
    size_t bytes = size * sizeof(int);
    h_idata = (int *)malloc(bytes);
    h_odata = (int *)malloc(grid.x * sizeof(int));
    temp = (int *)malloc(bytes);

    initializaData<int>(h_idata, size);
    memcpy(temp, h_idata, bytes);

    int *d_idata, *d_odata;
    ERROR_CHECK(cudaMalloc((void **)&d_idata, bytes));
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

    struct timeval tp;
    gettimeofday(&tp, NULL);
    double start_cpu = (double)(tp.tv_sec * 1.e3) + (double)(tp.tv_usec * 1.e-3);
    cpu_sum = recursiveReduce(temp, size);
    gettimeofday(&tp, NULL);
    double elapsed_time_cpu = (double)(tp.tv_sec * 1.e3) + (double)(tp.tv_usec * 1.e-3) - start_cpu;
    printf("cpu reduce\telapsed %g ms\tcpu_sum: %d\n", elapsed_time_cpu, cpu_sum);

    // 嵌套归约
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    gpuRecursiveReduce<<<grid, block>>>(d_idata, d_odata, block.x);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu nested\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

    memset(h_odata, 0, grid.x * sizeof(int));

    // 去除同步的嵌套归约
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    gpuRecursiveReduceNosync<<<grid, block>>>(d_idata, d_odata, block.x);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu nestedNosyn\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

    // 优化资源利用后的嵌套归约
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    gpuRecursiveReduce2<<<grid, block>>>(d_idata, d_odata, block.x / 2, block.x);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu nested2\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

    // 相邻匹配的并行归约求和
    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu neighbored\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

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
