if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "This project is intended to be built as part of LLVM via "
    "-DLLVM_EXTERNAL_PROJECTS=substrait_mlir"
    "-DLLVM_EXTERNAL_SUBSTRAIT_MLIR_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

################################################################################
# CMake options specific to this sub-project
################################################################################

option(SUBSTRAIT_MLIR_COMPILE_WARNING_AS_ERROR
       "value for 'CMAKE_COMPILE_WARNING_AS_ERROR' for Substrait MLIR targets"
       OFF)

################################################################################
# Set up dependencies
################################################################################

# Required for Substrait. v3.6.1 provided by Ubuntu 20.04 did not work due to
# an incompatibility with `-fno-rtti`, which LLVM uses, while v3.12.4 provided
# by Ubuntu 22.04 worked. Possibly some versions inbetween work as well.
find_package(Protobuf 3.12.0 REQUIRED)

################################################################################
# Set some variables
################################################################################
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)

list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)
set(MLIR_TABLEGEN_EXE mlir-tblgen)

set(SUBSTRAIT_MLIR_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SUBSTRAIT_MLIR_INCLUDE_DIR ${SUBSTRAIT_MLIR_MAIN_SRC_DIR}/include)
set(SUBSTRAIT_MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SUBSTRAIT_MLIR_TABLEGEN_OUTPUT_DIR ${SUBSTRAIT_MLIR_BINARY_DIR}/include)
message(STATUS "Substrait MLIR build directory: ${SUBSTRAIT_MLIR_BINARY_DIR}")

################################################################################
# Set include paths
################################################################################
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

include_directories(${SUBSTRAIT_MLIR_INCLUDE_DIR})
include_directories(${SUBSTRAIT_MLIR_TABLEGEN_OUTPUT_DIR})

################################################################################
# Enable python (assumes enabled MLIR bindings via MLIR_ENABLE_BINDINGS_PYTHON)
################################################################################
if(NOT DEFINED MLIR_ENABLE_BINDINGS_PYTHON)
  message(FATAL_ERROR
    "This project requires MLIR_ENABLE_BINDINGS_PYTHON=ON")
endif()

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

################################################################################
# Enable LLVM stuff
################################################################################
include(TableGen)
include(AddLLVM)
include(AddMLIR)

################################################################################
# Dependencies from git sub-modules
################################################################################
add_subdirectory(third_party)

################################################################################
# Subdirs to recurse into
################################################################################
add_custom_target(substrait-mlir-all)

# Set default value for COMPILE_WARNING_AS_ERROR for our targets.
set(CMAKE_COMPILE_WARNING_AS_ERROR ${SUBSTRAIT_MLIR_COMPILE_WARNING_AS_ERROR})

add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(python)
add_subdirectory(test)
add_subdirectory(tools)
