add_mlir_dialect(SubstraitOps substrait)
add_dependencies(MLIRSubstraitDialect MLIRSubstraitOpsIncGen)

# Add Enums
set(LLVM_TARGET_DEFINITIONS SubstraitOps.td)
mlir_tablegen(SubstraitEnums.h.inc -gen-enum-decls)
mlir_tablegen(SubstraitEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSubstraitEnumsIncGen)
add_dependencies(MLIRSubstraitDialect MLIRSubstraitEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS SubstraitInterfaces.td)
mlir_tablegen(SubstraitOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SubstraitOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(SubstraitTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(SubstraitTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRSubstraitInterfacesIncGen)
add_dependencies(MLIRSubstraitDialect MLIRSubstraitInterfacesIncGen)

add_dependencies(mlir-headers
  MLIRSubstraitOpsIncGen
)
