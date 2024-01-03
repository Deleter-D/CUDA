#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

__global__ void floatComputing(float* outputs, float* inputs, int size, size_t iterations)
{
    unsigned int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int threads_count = gridDim.x * blockDim.x;

    for (; tid < size; tid += threads_count)
    {
        float val = inputs[tid];

        for (size_t i = 0; i < iterations; i++)
        {
            val = (val + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }

        outputs[tid] = val;
    }
}

__global__ void doubleComputing(double* outputs, double* inputs, int size, size_t iterations)
{
    unsigned int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int threads_count = gridDim.x * blockDim.x;

    for (; tid < size; tid += threads_count)
    {
        double val = inputs[tid];

        for (size_t i = 0; i < iterations; i++)
        {
            val = (val + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }

        outputs[tid] = val;
    }
}

inline double miliseconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec * 1.e3 + (double)tp.tv_usec * 1.e-3);
}

void floatTest(size_t size, int iterations,
               int blocksPerGrid, int threadsPerBlock,
               float* toDeviceTime, float* kernelTime, float* fromDeviceTime,
               float* sample, int sampleLength)
{
    int i;
    float *h_floatInputs, *h_floatOutputs;
    float *d_floatInputs, *d_floatOutputs;

    h_floatInputs  = (float*)malloc(sizeof(float) * size);
    h_floatOutputs = (float*)malloc(sizeof(float) * size);
    ERROR_CHECK(cudaMalloc((void**)&d_floatInputs, sizeof(float) * size));
    ERROR_CHECK(cudaMalloc((void**)&d_floatOutputs, sizeof(float) * size));

    for (i = 0; i < size; i++)
    {
        h_floatInputs[i] = (float)i;
    }

    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    ERROR_CHECK(cudaEventRecord(start));
    ERROR_CHECK(cudaMemcpy(d_floatInputs, h_floatInputs, sizeof(float) * size, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(toDeviceTime, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    floatComputing<<<blocksPerGrid, threadsPerBlock>>>(d_floatOutputs, d_floatInputs, size, iterations);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(kernelTime, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    ERROR_CHECK(cudaMemcpy(h_floatOutputs, d_floatOutputs, sizeof(float) * size, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(fromDeviceTime, start, stop));

    for (i = 0; i < sampleLength; i++)
    {
        sample[i] = h_floatOutputs[i];
    }

    ERROR_CHECK(cudaEventDestroy(start));
    ERROR_CHECK(cudaEventDestroy(stop));
    ERROR_CHECK(cudaFree(d_floatInputs));
    ERROR_CHECK(cudaFree(d_floatOutputs));
    free(h_floatInputs);
    free(h_floatOutputs);
}

void doubleTest(size_t size, int iterations,
                int blocksPerGrid, int threadsPerBlock,
                float* toDeviceTime, float* kernelTime, float* fromDeviceTime,
                double* sample, int sampleLength)
{
    int i;
    double *h_doubleInputs, *h_doubleOutputs;
    double *d_doubleInputs, *d_doubleOutputs;

    h_doubleInputs  = (double*)malloc(sizeof(double) * size);
    h_doubleOutputs = (double*)malloc(sizeof(double) * size);
    ERROR_CHECK(cudaMalloc((void**)&d_doubleInputs, sizeof(double) * size));
    ERROR_CHECK(cudaMalloc((void**)&d_doubleOutputs, sizeof(double) * size));

    for (i = 0; i < size; i++)
    {
        h_doubleInputs[i] = (double)i;
    }

    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    ERROR_CHECK(cudaEventRecord(start));
    ERROR_CHECK(cudaMemcpy(d_doubleInputs, h_doubleInputs, sizeof(double) * size, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(toDeviceTime, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    doubleComputing<<<blocksPerGrid, threadsPerBlock>>>(d_doubleOutputs, d_doubleInputs, size, iterations);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(kernelTime, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    ERROR_CHECK(cudaMemcpy(h_doubleOutputs, d_doubleOutputs, sizeof(double) * size, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(fromDeviceTime, start, stop));

    for (i = 0; i < sampleLength; i++)
    {
        sample[i] = h_doubleOutputs[i];
    }

    ERROR_CHECK(cudaEventDestroy(start));
    ERROR_CHECK(cudaEventDestroy(stop));
    ERROR_CHECK(cudaFree(d_doubleInputs));
    ERROR_CHECK(cudaFree(d_doubleOutputs));
    free(h_doubleInputs);
    free(h_doubleOutputs);
}

int main(int argc, char const* argv[])
{
    int i;
    double meanFloatToDeviceTime, meanFloatKernelTime, meanFloatFromDeviceTime;
    double meanDoubleToDeviceTime, meanDoubleKernelTime, meanDoubleFromDeviceTime;
    cudaDeviceProp deviceProperties;
    size_t totalMem, freeMem;
    float* floatSample;
    double* doubleSample;
    int sampleLength = 10;
    int nRuns        = 5;
    int nKernelIters = 20;

    meanFloatToDeviceTime = meanFloatKernelTime = meanFloatFromDeviceTime = 0.0;
    meanDoubleToDeviceTime = meanDoubleKernelTime = meanDoubleFromDeviceTime = 0.0;

    ERROR_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    ERROR_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    size_t size         = (freeMem * 0.9 / 2) / sizeof(double);
    int threadsPerBlock = 256;
    int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

    if (blocksPerGrid > deviceProperties.maxGridSize[0])
    {
        blocksPerGrid = deviceProperties.maxGridSize[0];
    }

    printf("Running %d blocks with %d threads/block over %lu elements\n", blocksPerGrid, threadsPerBlock, size);

    floatSample  = (float*)malloc(sizeof(float) * sampleLength);
    doubleSample = (double*)malloc(sizeof(double) * sampleLength);

    for (i = 0; i < nRuns; i++)
    {
        float toDeviceTime, kernelTime, fromDeviceTime;

        floatTest(size, nKernelIters,
                  blocksPerGrid, threadsPerBlock,
                  &toDeviceTime, &kernelTime, &fromDeviceTime,
                  floatSample, sampleLength);
        meanFloatToDeviceTime += toDeviceTime;
        meanFloatKernelTime += kernelTime;
        meanFloatFromDeviceTime += fromDeviceTime;

        doubleTest(size, nKernelIters,
                   blocksPerGrid, threadsPerBlock,
                   &toDeviceTime, &kernelTime, &fromDeviceTime,
                   doubleSample, sampleLength);
        meanDoubleToDeviceTime += toDeviceTime;
        meanDoubleKernelTime += kernelTime;
        meanDoubleFromDeviceTime += fromDeviceTime;

        if (i == 0)
        {
            int j;
            printf("Input\tDiff Between Single- and Double-Precision\n");
            printf("------\t-----------------------------------------\n");

            for (j = 0; j < sampleLength; j++)
            {
                printf("%d\t%.20e\n", j, fabs(doubleSample[j] - (double)floatSample[j]));
            }

            printf("\n");
        }
    }

    meanFloatToDeviceTime /= nRuns;
    meanFloatKernelTime /= nRuns;
    meanFloatFromDeviceTime /= nRuns;
    meanDoubleToDeviceTime /= nRuns;
    meanDoubleKernelTime /= nRuns;
    meanDoubleFromDeviceTime /= nRuns;

    printf("For single-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f ms\n", meanFloatToDeviceTime);
    printf("  Kernel execution: %f ms\n", meanFloatKernelTime);
    printf("  Copy from device: %f ms\n", meanFloatFromDeviceTime);
    printf("For double-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f ms (%.2fx slower than single-precision)\n", meanDoubleToDeviceTime, meanDoubleToDeviceTime / meanFloatToDeviceTime);
    printf("  Kernel execution: %f ms (%.2fx slower than single-precision)\n", meanDoubleKernelTime, meanDoubleKernelTime / meanFloatKernelTime);
    printf("  Copy from device: %f ms (%.2fx slower than single-precision)\n", meanDoubleFromDeviceTime, meanDoubleFromDeviceTime / meanFloatFromDeviceTime);

    return 0;
}
