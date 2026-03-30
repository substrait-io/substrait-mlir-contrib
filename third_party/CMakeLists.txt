# ##############################################################################
# Substrait
# ##############################################################################

# Some of our dependencies don't properyl IWYU, so they don't work on
# implementations of the standard headers that don't transitively include
# `cstdint` for them. Work-around: include it always via compiler flag.
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:
       $<$<CXX_COMPILER_ID:MSVC>:/FIcstdint>
       $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-include;cstdint>>")

# The way how Substrait uses sanitizers seems to be incompatible with some flags
# set by LLVM, leading to linker errors, so we deactivate them.
set(SUBSTRAIT_CPP_SANITIZE_DEBUG_BUILD OFF)

# LLVM sets this to `ON`, leading to a compilation error due to a file called
# `time.h` in one of Abseil's folders.
set(CMAKE_INCLUDE_CURRENT_DIR OFF)

if(${LLVM_INCLUDE_TESTS})
  message(
    FATAL_ERROR
      "LLVM_INCLUDE_TESTS is enabled but breaks the build due "
      "to a target name clash with 'abseil-cpp'. Please run "
      "cmake again with '-DLLVM_INCLUDE_TESTS=OFF'.")
endif()

# Propagate zlib settings to protobuf build.
set(protobuf_WITH_ZLIB
    ${LLVM_ENABLE_ZLIB}
    CACHE BOOL "" FORCE)

# Add `substrait-cpp` as a subdirectory with above settings.
set(SUBSTRAIT_CPP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/substrait-cpp)
add_subdirectory(${SUBSTRAIT_CPP_ROOT_DIR})
