find_package(OpenMP REQUIRED)

add_executable(01_hyper-Q_depth 01_hyper-Q_depth.cu)
set_target_properties(01_hyper-Q_depth PROPERTIES CUDA_ARCHITECTURES ${TARGET_ARCHITECTURES})
install(TARGETS 01_hyper-Q_depth DESTINATION 05_stream_and_concurrence)

add_executable(02_hyper-Q_breadth 02_hyper-Q_breadth.cu)
set_target_properties(02_hyper-Q_breadth PROPERTIES CUDA_ARCHITECTURES ${TARGET_ARCHITECTURES})
install(TARGETS 02_hyper-Q_breadth DESTINATION 05_stream_and_concurrence)

add_executable(03_hyper-Q_OpenMP 03_hyper-Q_OpenMP.cu)
set_target_properties(03_hyper-Q_OpenMP PROPERTIES CUDA_ARCHITECTURES ${TARGET_ARCHITECTURES})
set_target_properties(03_hyper-Q_OpenMP PROPERTIES COMPILE_FLAGS "-O3 -Xcompiler -fopenmp -lgomp")
target_link_libraries(03_hyper-Q_OpenMP OpenMP::OpenMP_CXX)
install(TARGETS 03_hyper-Q_OpenMP DESTINATION 05_stream_and_concurrence)

add_executable(04_hyper-Q_dependence 04_hyper-Q_dependence.cu)
set_target_properties(04_hyper-Q_dependence PROPERTIES CUDA_ARCHITECTURES ${TARGET_ARCHITECTURES})
install(TARGETS 04_hyper-Q_dependence DESTINATION 05_stream_and_concurrence)