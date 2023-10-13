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
    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

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

int main(int argc, char const *argv[])
{
    setDevice();

    int size = 1 << 24;
    printf("Array Size: %d\n", size);

    int cpu_sum, gpu_sum;

    dim3 block(512);
    dim3 grid((size + block.x - 1) / block.x);

    int *h_idata, *h_odata, *temp;
    h_idata = (int *)malloc(size * sizeof(int));
    h_odata = (int *)malloc(grid.x * sizeof(int));
    temp = (int *)malloc(size * sizeof(int)); // 用于CPU端求和

    initializaData<int>(h_idata, size);
    memcpy(temp, h_idata, size * sizeof(int));

    int *d_idata, *d_odata;
    ERROR_CHECK(cudaMalloc((void **)&d_idata, size * sizeof(int)));
    ERROR_CHECK(cudaMalloc((void **)&d_odata, size * sizeof(int)));

    ERROR_CHECK(cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));

    // 预热
    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));
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
    ERROR_CHECK(cudaEventRecord(start));
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    ERROR_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu neighbored\telapsed %g ms\tgpu_sum: %d\t<<<%d, %d>>>\n", elapsedTime, gpu_sum, grid.x, block.x);

    ERROR_CHECK(cudaFree(d_idata));
    ERROR_CHECK(cudaFree(d_odata));

    free(h_idata);
    free(h_odata);
    free(temp);

    cudaDeviceReset();

    if (cpu_sum == gpu_sum)
        printf("Result correct!\n");
    else
        printf("Result error!\n");

    return 0;
}
