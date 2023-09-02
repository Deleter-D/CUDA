#include <stdio.h>
#include <stdlib.h>

/*
    GPU算子开发基本思路

    分为两大部分，主机代码和设备代码

    设备代码
        - 每个线程实际的运算逻辑

    主机代码
        - 初始化GPU设备
        - 主机内存分配及其初始化
        - 设备内存分配及其初始化
        - 主机到设备的数据般移（将待计算数据般移到设备）
        - 调用核函数（执行计算过程）
        - 设备到主机的数据般移（将计算结果般移到主机）
        - 设备内存释放
        - 主机内存释放
*/

void dataInit(float *ptr, int elemCount)
{
    for (int i = 0; i < elemCount; ++i)
    {
        // 随即生成-100到100之间的浮点数
        ptr[i] = (float)(rand()) / RAND_MAX * 200 - 100;
    }
}

// 设备代码只能由设备调用
__device__ float add(const float x, const float y)
{
    return x + y;
}

__global__ void vecAdd(float *A, float *B, float *C, const int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = bid * blockDim.x + tid;

    // 由于为了适应总线程数无法整除block数，会多申请一个block
    // 而总的线程数就会大于元素总个数，所以需要一个判断来控制运算的合法性
    if (id < N)
        C[id] = add(A[id], B[id]);
}

int main(int argc, char const *argv[])
{
    // >>> 初始化GPU设备 <<<
    int deviceCount = 0;
    cudaError_t flag = cudaGetDeviceCount(&deviceCount); // 获取设备数量
    // 该API返回一个cudaError_t的枚举类
    if (flag != cudaError_t::cudaSuccess | deviceCount == 0)
    {
        printf("Get device count faild. There is no device in your computer.\n");
        exit(-1);
    }
    else
    {
        printf("Get device count successfully.\n");
        printf("There %s %d device%s in your computer.\n", (deviceCount > 1 ? "are" : "is"), deviceCount, (deviceCount > 1 ? "s" : ""));
    }

    int device = 0;
    flag = cudaSetDevice(device); // 设置执行设备代码的目标设备
    if (flag != cudaSuccess)
    {
        printf("Fail to set GPU %d for computing.\n", device);
        exit(-1);
    }
    else
    {
        printf("Set GPU %d for computing.\n", device);
    }
    // 上面的代码可以封装到一个头文件中，方便日后使用，见utils/common.cuh

    // >>> 主机内存分配及其初始化 <<<
    int elemCount = 513;                           // 元素个数
    size_t bytesCount = elemCount * sizeof(float); // 总字节数
    float *h_A, *h_B, *h_C;
    // 申请主机内存
    h_A = (float *)malloc(bytesCount);
    h_B = (float *)malloc(bytesCount);
    h_C = (float *)malloc(bytesCount);
    // 初始化
    if (h_A != NULL && h_B != NULL && h_C != NULL)
    {
        memset(h_A, 0, bytesCount);
        memset(h_B, 0, bytesCount);
        memset(h_C, 0, bytesCount);
    }
    else
    {
        printf("Fail to allocate host memory.\n");
        exit(-1);
    }

    // >>> 设备内存分配及其初始化 <<<
    float *d_A, *d_B, *d_C;
    // 申请设备内存
    cudaMalloc((void **)&d_A, bytesCount);
    cudaMalloc((void **)&d_B, bytesCount);
    cudaMalloc((void **)&d_C, bytesCount);
    // 初始化
    if (d_A != NULL && d_B != NULL && d_C != NULL)
    {
        cudaMemset(d_A, 0, bytesCount);
        cudaMemset(d_B, 0, bytesCount);
        cudaMemset(d_C, 0, bytesCount);
    }
    else
    {
        printf("Fail to allocate device memory.\n");
        exit(-1);
    }

    // 初始化主机端数据
    srand(42);
    dataInit(h_A, elemCount);
    dataInit(h_B, elemCount);

    // >>> 主机到设备的数据般移（将待计算数据般移到设备）<<<
    cudaMemcpy(d_A, h_A, bytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesCount, cudaMemcpyHostToDevice);

    // >>> 调用核函数（执行计算过程）<<<
    dim3 blocksize(32);                                // 每个block设置32个thread
    dim3 gridsize((elemCount + blocksize.x - 1) / 32); // 防止元素个数不是32的整倍数
    vecAdd<<<gridsize, blocksize>>>(d_A, d_B, d_C, elemCount);
    // 同步API，等待核函数计算完成
    // cudaDeviceSynchronize();
    // 这里可以不调用同步函数，因为下方有cudaMemcpy，会隐式同步

    // >>> 设备到主机的数据般移（将计算结果般移到主机）<<<
    cudaMemcpy(h_C, d_C, bytesCount, cudaMemcpyDeviceToHost);

    // 输出前10个计算结果
    for (int i = 0; i < 10; i++)
    {
        printf("vector A: %.2f\tvector B: %.2f\tresult: %.2f\n", h_A[i], h_B[i], h_C[i]);
    }

    // >>> 设备内存释放 <<<
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // >>> 主机内存释放 <<<
    free(h_A);
    free(h_B);
    free(h_C);

    cudaDeviceReset();

    return 0;
}