add_executable(01_hyper-Q 01_hyper-Q.cu)
set_target_properties(01_hyper-Q PROPERTIES CUDA_ARCHITECTURES ${TARGET_ARCHITECTURES})
install(TARGETS 01_hyper-Q DESTINATION 05_stream_and_concurrence)