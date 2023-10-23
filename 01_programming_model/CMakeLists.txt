add_executable(01_hello_world 01_hello_world.cu)
set_target_properties(01_hello_world PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 01_hello_world DESTINATION 01_programming_model)

add_executable(02_error_handle 02_error_handle.cu)
set_target_properties(02_error_handle PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 02_error_handle DESTINATION 01_programming_model)

add_executable(03_profiling 03_profiling.cu)
set_target_properties(03_profiling PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 03_profiling DESTINATION 01_programming_model)

add_executable(04_threads_management 04_threads_management.cu)
set_target_properties(04_threads_management PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 04_threads_management DESTINATION 01_programming_model)

add_executable(05_memory_management 05_memory_management.cu)
set_target_properties(05_memory_management PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 05_memory_management DESTINATION 01_programming_model)

add_executable(06_example_sum_matrix_2D-grid_2D-block 06_example_sum_matrix_2D-grid_2D-block.cu)
set_target_properties(06_example_sum_matrix_2D-grid_2D-block PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 06_example_sum_matrix_2D-grid_2D-block DESTINATION 01_programming_model)

add_executable(07_example_sum_matrix_1D-grid_1D-block 07_example_sum_matrix_1D-grid_1D-block.cu)
set_target_properties(07_example_sum_matrix_1D-grid_1D-block PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 07_example_sum_matrix_1D-grid_1D-block DESTINATION 01_programming_model)

add_executable(08_example_sum_matrix_2D-grid_1D-block 08_example_sum_matrix_2D-grid_1D-block.cu)
set_target_properties(08_example_sum_matrix_2D-grid_1D-block PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 08_example_sum_matrix_2D-grid_1D-block DESTINATION 01_programming_model)

add_executable(09_device_management 09_device_management.cu)
set_target_properties(09_device_management PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 09_device_management DESTINATION 01_programming_model)