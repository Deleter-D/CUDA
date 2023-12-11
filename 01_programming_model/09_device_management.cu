#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    // >>>>> 运行时API获取GPU信息 >>>>>
    int deviceCount      = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount); // 获取设备数量
    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        int driverVersion = 0, runtimeVersion = 0;
        // 指定运行设备
        cudaSetDevice(dev);
        // 获取设备属性结构体
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        // 获取驱动版本
        cudaDriverGetVersion(&driverVersion);
        // 获取运行时版本
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        // 获取设备计算能力
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
        // 获取设备全局内存容量
        printf("  Total amount of global memory                  %.2f GBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem / (pow(1024.0, 3)), (unsigned long long)deviceProp.totalGlobalMem);
        // 获取GPU时钟频率
        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        // 获取内存频率
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        // 获取内存总线位宽
        printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);
        // 若GPU有L2缓存，则获取其容量
        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %.2f MBytes (%d bytes)\n", (float)deviceProp.l2CacheSize / (pow(1024.0, 2)), deviceProp.l2CacheSize);
        }
        // 获取各个维度的最大纹理尺寸
        printf("  Max Texture Dimension Size (x, y, z)           1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n", deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        // 获取各个维度的最大分层纹理尺寸
        printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d, %d) x %d\n", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
        // 获取常量内存容量
        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        // 获取每个block的共享内存容量
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        // 获取每个block可用的32位寄存器总数
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        // 获取线程束大小
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        // 获取每个多处理器的最大驻留线程数
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        // 获取每个block支持的最大线程数
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        // 获取各个维度的block支持的最大线程数
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        // 获取各个维度的grid支持的最大block数
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        // 获取内存拷贝操作允许的最大间距（字节）
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
    }
    // <<<<< 运行时API获取GPU信息 <<<<<

    // >>>>> 确定最优GPU >>>>>
    if (deviceCount > 1)
    {
        int maxMultiprocessors = 0, maxDevice = 0;
        for (int device = 0; device < deviceCount; device++)
        {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            if (maxMultiprocessors < props.multiProcessorCount)
            {
                maxMultiprocessors = props.multiProcessorCount;
                maxDevice          = device;
            }
        }
        cudaSetDevice(maxDevice);
    }
    // <<<<< 确定最优GPU <<<<<

    exit(EXIT_SUCCESS);
}
