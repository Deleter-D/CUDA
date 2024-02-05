#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

// 调用主机端API在主机生成
void generate_on_host()
{
    int n = 10;

    const float mean   = 1.0f;
    const float stddev = 2.0f;

    float *data = (float *)malloc(n * sizeof(float));

    // 准备生成器、RNG类型、排序方式、偏移量、种子
    curandGenerator_t gen;
    curandRngType_t rng             = CURAND_RNG_PSEUDO_XORWOW;
    curandOrdering_t order          = CURAND_ORDERING_PSEUDO_BEST;
    const unsigned long long offset = 0ULL;
    const unsigned long long seed   = 1234ULL;

    // 创建主机端生成器，并指定RNG类型
    ERROR_CHECK_CURAND(curandCreateGeneratorHost(&gen, rng));

    // 设置偏移量
    ERROR_CHECK_CURAND(curandSetGeneratorOffset(gen, offset));

    // 设置排序方式
    ERROR_CHECK_CURAND(curandSetGeneratorOrdering(gen, order));

    // 设置种子
    ERROR_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));

    // 在主机端生成，以正态分布为例
    ERROR_CHECK_CURAND(curandGenerateNormal(gen, data, n, mean, stddev));
    // 此外还有均匀分布、对数正态分布和泊松分布

    printf("%d float number generated on host:\n\t", n);
    for (int i = 0; i < n; i++)
    {
        printf("%f ", data[i]);
    }
    printf("\n");
}

// 调用主机端API在设备生成
void generate_on_device()
{
    int n = 10;

    const float mean   = 1.0f;
    const float stddev = 2.0f;

    float *data = (float *)malloc(n * sizeof(float));
    float *d_data;
    ERROR_CHECK(cudaMalloc((void **)&d_data, n * sizeof(float)));

    // 准备生成器、RNG类型、排序方式、偏移量、种子
    curandGenerator_t gen;
    curandRngType_t rng             = CURAND_RNG_PSEUDO_XORWOW;
    curandOrdering_t order          = CURAND_ORDERING_PSEUDO_BEST;
    const unsigned long long offset = 0ULL;
    const unsigned long long seed   = 1234ULL;

    // 创建设备端生成器，并指定RNG类型
    ERROR_CHECK_CURAND(curandCreateGenerator(&gen, rng));

    // 设置偏移量
    ERROR_CHECK_CURAND(curandSetGeneratorOffset(gen, offset));

    // 设置排序方式
    ERROR_CHECK_CURAND(curandSetGeneratorOrdering(gen, order));

    // 设置种子
    ERROR_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));

    // 在主机端生成，以正态分布为例
    ERROR_CHECK_CURAND(curandGenerateNormal(gen, d_data, n, mean, stddev));
    // 此外还有均匀分布、对数正态分布和泊松分布

    // 将生成结果搬移回主机
    ERROR_CHECK(cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());

    printf("%d float number generated on device:\n\t", n);
    for (int i = 0; i < n; i++)
    {
        printf("%f ", data[i]);
    }
    printf("\n");
}

// 调用设备端API
__global__ void generate_in_kernel()
{
    // 针对不同RNG算法创建状态
    curandStateXORWOW_t rand_state;
    // 为保证随机性，每个线程应当设置不同的种子，且避免用当前时间戳作为种子
    unsigned long long seed = threadIdx.x;
    // 子序列会使得curand_init()返回的序列是调用了(2^67 * subsequence + offset)次curand()的结果
    unsigned long long subsequence = 1ULL;
    unsigned long long offset      = 0ULL;

    curand_init(seed, subsequence, offset, &rand_state);

    float x = curand_normal(&rand_state);
    printf("generate in kernel by thread %d:\t%f\n", threadIdx.x, x);
}

int main(int argc, char const *argv[])
{
    setDevice();

    generate_on_host();

    generate_on_device();

    generate_in_kernel<<<1, 10>>>();

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
