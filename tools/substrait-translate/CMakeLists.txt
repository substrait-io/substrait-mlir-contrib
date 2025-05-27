set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS )
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(substrait-translate
  substrait-translate.cpp
  )
add_dependencies(substrait-mlir-tools substrait-translate)

target_link_libraries(substrait-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  LLVMSupport
  MLIRIR
  MLIRTargetSubstraitPB
  MLIRTranslateLib
  MLIRSupport
  )

mlir_check_all_link_libraries(substrait-translate)
