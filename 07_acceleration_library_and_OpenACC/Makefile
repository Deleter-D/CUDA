CC = pgcc
CXX = pgc++
CFLAGS = -acc -Minfo=accel
CUDA_INCLUDE_PATH = /usr/local/cuda/include/
CUDA_LIB_PATH = /usr/local/cuda/lib64/
OUTPUT_PATH = ../executable/07_acceleration_library_and_OpenACC

all: 05_openacc_kernels 06_openacc_parallel 07_openacc_data 08_cuda_openacc

05_openacc_kernels: 05_openacc_kernels.c
	$(CC) $(CFLAGS) -o $(OUTPUT_PATH)/05_openacc_kernels $<

06_openacc_parallel: 06_openacc_parallel.c
	$(CC) $(CFLAGS) -o $(OUTPUT_PATH)/06_openacc_parallel $<

07_openacc_data: 07_openacc_data.c
	$(CC) $(CFLAGS) -o $(OUTPUT_PATH)/07_openacc_data $<

08_cuda_openacc: 08_cuda_openacc.cu
	$(CXX) $(CFLAGS) -I $(CUDA_INCLUDE_PATH) -L $(CUDA_LIB_PATH) -lcurand -lcublas -lcudart -o $(OUTPUT_PATH)/08_cuda_openacc $<


clean:
	rm -f $(OUTPUT_PATH)/05_openacc_kernels
	rm -f $(OUTPUT_PATH)/06_openacc_parallel
	rm -f $(OUTPUT_PATH)/07_openacc_data
	rm -f $(OUTPUT_PATH)/08_cuda_openacc
