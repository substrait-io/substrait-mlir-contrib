get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(substrait-opt
        substrait-opt.cpp
  )
add_dependencies(substrait-mlir-tools substrait-opt)

set(test_libs
)

target_link_libraries(substrait-opt
  PRIVATE
  ${conversion_libs}
  ${extension_libs}
  ${dialect_libs}
  ${test_libs}
  MLIRIR
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRSupport
)

mlir_check_all_link_libraries(substrait-opt)
