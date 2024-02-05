#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

#include <complex>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

// 归一化数据
__global__ void scaling_kernel(cufftComplex *data, int element_count, float scale)
{
    const int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (auto i = tid; i < element_count; i += stride)
    {
        data[tid].x *= scale;
        data[tid].y *= scale;
    }
}

int main(int argc, char const *argv[])
{
    setDevice();

    int fft_size      = 8;
    int batch_size    = 2;
    int element_count = fft_size * batch_size;

    using data_t = std::complex<float>;

    data_t *data = (data_t *)malloc(element_count * sizeof(data_t));

    for (int i = 0; i < element_count; i++)
    {
        data[i] = data_t(i, -i);
    }

    cufftComplex *d_data;
    ERROR_CHECK(cudaMalloc((void **)&d_data, element_count * sizeof(data_t)));
    ERROR_CHECK(cudaMemcpy(d_data, data, element_count * sizeof(data_t), cudaMemcpyHostToDevice));

    // 创建cuFFT句柄
    cufftHandle plan;
    ERROR_CHECK_CUFFT(cufftCreate(&plan));

    // 创建cuFFT的plan
    ERROR_CHECK_CUFFT(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch_size));

    // 执行正向变换
    ERROR_CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    // 归一化
    scaling_kernel<<<1, 128>>>(d_data, element_count, 1.f / fft_size);

    // 执行逆向变换
    ERROR_CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    // 销毁handel
    ERROR_CHECK_CUFFT(cufftDestroy(plan));

    ERROR_CHECK(cudaMemcpy(data, d_data, element_count * sizeof(data_t), cudaMemcpyDeviceToHost));

    // 资源释放
    ERROR_CHECK(cudaFree(d_data));

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
