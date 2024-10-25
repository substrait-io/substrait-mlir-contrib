add_custom_target(substrait-mlir-tools)
add_dependencies(substrait-mlir-all substrait-mlir-tools)

add_subdirectory(substrait-lsp-server)
add_subdirectory(substrait-opt)
add_subdirectory(substrait-translate)
