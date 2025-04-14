cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(SEGREDUCE
    VERSION 1.0
    LANGUAGES CXX CUDA
)


find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
# find_package(nanobind CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dependencies/moderngpu)
# nanobind_add_module(pysegreduce
#     ext.cu
# )
pybind11_add_module(pysegreduce
    ext.cu
)
target_link_libraries(pysegreduce PRIVATE MODERNGPU)

# print python library directory
# message(STATUS "Python_SITELIB: ${Python_SITELIB}")

install(TARGETS pysegreduce
    DESTINATION ${Python_SITELIB}
)

