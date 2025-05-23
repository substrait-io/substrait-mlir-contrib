add_mlir_dialect_library(MLIRSubstraitDialect
  Substrait.cpp

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRCastInterfaces
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSupport

  DEPENDS
  MLIRSubstraitIncGen
)
