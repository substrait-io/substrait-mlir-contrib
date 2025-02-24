# Add dialect, types, and ops.
add_mlir_dialect(SubstraitOps substrait)

# Add enums.
set(LLVM_TARGET_DEFINITIONS SubstraitOps.td)
mlir_tablegen(SubstraitEnums.h.inc -gen-enum-decls)
mlir_tablegen(SubstraitEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSubstraitEnumsIncGen)

# Add attributes.
set(LLVM_TARGET_DEFINITIONS SubstraitAttrs.td)
mlir_tablegen(SubstraitOpsAttrs.h.inc --gen-attrdef-decls)
mlir_tablegen(SubstraitOpsAttrs.cpp.inc --gen-attrdef-defs)
add_public_tablegen_target(MLIRSubstraitAttrsIncGen)

# Add interfaces.
set(LLVM_TARGET_DEFINITIONS SubstraitInterfaces.td)
mlir_tablegen(SubstraitAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(SubstraitAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(SubstraitOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SubstraitOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(SubstraitTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(SubstraitTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRSubstraitInterfacesIncGen)

add_custom_target(MLIRSubstraitIncGen
  DEPENDS
  MLIRSubstraitOpsIncGen
  MLIRSubstraitEnumsIncGen
  MLIRSubstraitAttrsIncGen
  MLIRSubstraitInterfacesIncGen
)
