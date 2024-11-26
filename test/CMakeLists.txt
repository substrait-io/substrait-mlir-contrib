llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  )

# Passed to lit.site.cfg.py.in to set up the path where to find libraries.
set(MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(SUBSTRAIT_MLIR_TEST_DEPENDS
  count
  FileCheck
  substrait-opt
  substrait-translate
  mlir_async_runtime
  mlir-cpu-runner
  mlir_c_runner_utils
  mlir_runner_utils
  not
  )
list(APPEND SUBSTRAIT_MLIR_TEST_DEPENDS SubstraitMLIRPythonModules)

add_lit_testsuite(check-substrait-mlir "Running the Substrait MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SUBSTRAIT_MLIR_TEST_DEPENDS}
  )
set_target_properties(check-substrait-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(
  "Substrait-MLIR" ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${SUBSTRAIT_MLIR_TEST_DEPENDS}
  )

add_custom_target(check-substrait-mlir-dependencies)
add_dependencies(check-substrait-mlir-dependencies ${SUBSTRAIT_MLIR_TEST_DEPENDS})
add_dependencies(substrait-mlir-all check-substrait-mlir-dependencies)
