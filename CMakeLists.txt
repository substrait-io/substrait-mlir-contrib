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
# Set up clang-tidy
################################################################################

option(SUBSTRAIT_MLIR_ENABLE_CLANG_TIDY
       "Enable clang-tidy static analysis on Substrait MLIR targets"
       OFF)
option(SUBSTRAIT_MLIR_CLANG_TIDY_PATH "path to 'clang-tidy' executable")

if(SUBSTRAIT_MLIR_ENABLE_CLANG_TIDY)
  # Find `clang-tidy` executable.
  get_filename_component(CLANG_TIDY_NAME
    "${SUBSTRAIT_MLIR_CLANG_TIDY_PATH}" NAME)
  get_filename_component(CLANG_TIDY_DIR
    "${SUBSTRAIT_MLIR_CLANG_TIDY_PATH}" DIRECTORY)
  find_program(CLANG_TIDY_EXE
    HINTS ${CLANG_TIDY_DIR}
    NAMES "clang-tidy" "${CLANG_TIDY_NAME}"
    REQUIRED)
  if("${CLANG_TIDY_EXE}" STREQUAL "")
    message(FATAL_ERROR "failed to find clang-tidy executable")
  endif()

  # Assemble command line.
  set(CLANG_TIDY_CONFIG_FILE "${SUBSTRAIT_MLIR_MAIN_SRC_DIR}/.clang-tidy")
  set(CLANG_TIDY_COMMAND
    "${CLANG_TIDY_EXE};--config-file=${CLANG_TIDY_CONFIG_FILE}")
  if (${SUBSTRAIT_MLIR_COMPILE_WARNING_AS_ERROR})
    set(CLANG_TIDY_COMMAND "${CLANG_TIDY_COMMAND};--warnings-as-errors=*")
  endif()
  set(CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR
    "${SUBSTRAIT_MLIR_BINARY_DIR}/clang-tidy-fixes")
  message(STATUS "clang-tidy enabled, command: ${CLANG_TIDY_COMMAND}")
endif(SUBSTRAIT_MLIR_ENABLE_CLANG_TIDY)

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
# Dependencies from git sub-modules
################################################################################
add_subdirectory(third_party)

################################################################################
# Subdirs to recurse into
################################################################################
add_custom_target(substrait-mlir-all)

# Set default value for COMPILE_WARNING_AS_ERROR for our targets.
set(CMAKE_COMPILE_WARNING_AS_ERROR ${SUBSTRAIT_MLIR_COMPILE_WARNING_AS_ERROR})

# Enable clang-tidy for our targets.
set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_COMMAND})

add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(python)
add_subdirectory(test)
add_subdirectory(tools)

# Disable clang-tidy for everything that may came after.
unset(CMAKE_CXX_CLANG_TIDY)
