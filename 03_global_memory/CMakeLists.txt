add_executable(01_global_variable 01_global_variable.cu)
set_target_properties(01_global_variable PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS  DESTINATION 03_global_memory)

add_executable(02_pageable_memory 02_pageable_memory.cu)
set_target_properties(02_pageable_memory PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 02_pageable_memory DESTINATION 03_global_memory)

add_executable(03_page_locked_memory 03_page_locked_memory.cu)
set_target_properties(03_page_locked_memory PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 03_page_locked_memory DESTINATION 03_global_memory)

add_executable(04_mapped_memory 04_mapped_memory.cu)
set_target_properties(04_mapped_memory PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 04_mapped_memory DESTINATION 03_global_memory)

add_executable(05_unified_virtual_address 05_unified_virtual_address.cu)
set_target_properties(05_unified_virtual_address PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 05_unified_virtual_address DESTINATION 03_global_memory)

add_executable(06_read_segment 06_read_segment.cu)
set_target_properties(06_read_segment PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 06_read_segment DESTINATION 03_global_memory)

add_executable(07_write_segment 07_write_segment.cu)
set_target_properties(07_write_segment PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 07_write_segment DESTINATION 03_global_memory)

add_executable(08_array_of_structure 08_array_of_structure.cu)
set_target_properties(08_array_of_structure PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 08_array_of_structure DESTINATION 03_global_memory)

add_executable(09_structure_of_array 09_structure_of_array.cu)
set_target_properties(09_structure_of_array PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 09_structure_of_array DESTINATION 03_global_memory)

add_executable(10_matrix_transpose 10_matrix_transpose.cu)
set_target_properties(10_matrix_transpose PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 10_matrix_transpose DESTINATION 03_global_memory)

add_executable(11_matrix_sum_managed 11_matrix_sum_managed.cu)
set_target_properties(11_matrix_sum_managed PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 11_matrix_sum_managed DESTINATION 03_global_memory)

add_executable(12_matrix_sum_manual 12_matrix_sum_manual.cu)
set_target_properties(12_matrix_sum_manual PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS 12_matrix_sum_manual DESTINATION 03_global_memory)