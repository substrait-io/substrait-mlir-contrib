include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `substrait_mlir`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=substrait_mlir.")

# ###############################################################################
# Sources
# ###############################################################################
declare_mlir_python_sources(SubstraitMLIRPythonSources)
declare_mlir_python_sources(SubstraitMLIRPythonSources.Dialects
  ADD_TO_PARENT SubstraitMLIRPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SubstraitMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/substrait_mlir"
  TD_FILE dialects/SubstraitOps.td
  SOURCES
  dialects/substrait.py
  DIALECT_NAME substrait
)

declare_mlir_python_extension(SubstraitMLIRPythonSources.DialectExtension
  MODULE_NAME _substraitDialects
  ADD_TO_PARENT SubstraitMLIRPythonSources
  SOURCES
  SubstraitDialects.cpp
  EMBED_CAPI_LINK_LIBS
  SubstraitMLIRCAPI
)

declare_mlir_python_sources(SubstraitMLIRPythonSources.MLIRLibs
  ADD_TO_PARENT SubstraitMLIRPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/substrait_mlir"
  SOURCES
    _mlir_libs/_substraitDialects/__init__.pyi
    _mlir_libs/_substraitDialects/substrait.pyi
)

# ###############################################################################
# Common CAPI
# ###############################################################################
add_mlir_python_common_capi_library(SubstraitMLIRPythonCAPI
  INSTALL_COMPONENT SubstraitMLIRPythonModules
  INSTALL_DESTINATION python_packages/substrait_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${SUBSTRAIT_MLIR_BINARY_DIR}/python_packages/substrait_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
  SubstraitMLIRPythonSources
  MLIRPythonExtension.RegisterEverything
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects
  MLIRPythonSources.ExecutionEngine
)

# ###############################################################################
# Instantiation of all Python modules
# ###############################################################################
add_mlir_python_modules(SubstraitMLIRPythonModules
  ROOT_PREFIX "${SUBSTRAIT_MLIR_BINARY_DIR}/python_packages/substrait_mlir"
  INSTALL_PREFIX "python_packages/substrait_mlir"
  DECLARED_SOURCES
  SubstraitMLIRPythonSources
  LLVMSupport
  MLIRPythonExtension.RegisterEverything
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects
  MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
  SubstraitMLIRPythonCAPI
)

add_dependencies(substrait-mlir-all SubstraitMLIRPythonModules)
