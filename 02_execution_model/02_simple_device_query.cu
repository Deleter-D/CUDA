#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char const *argv[])
{
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device %d: %s\n", device, prop.name);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Total amount of constant memory: %4.2f KB\n", prop.totalConstMem / 1024.0);
    printf("Total amount of shared memory per block: %4.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Total amount of registers available per block: %d\n", prop.regsPerBlock);

    printf("Warp size: %d\n", prop.warpSize);
    printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);

    return 0;
}
