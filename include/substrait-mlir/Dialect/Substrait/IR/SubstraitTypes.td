//===-- SubstraitTypes.td - Substrait dialect types --------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITTYPES
#define SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITTYPES

include "substrait-mlir/Dialect/Substrait/IR/SubstraitDialect.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"

// Base class for Substrait dialect types.
class Substrait_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Substrait_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

/// Currently supported atomic types. These correspond directly to the types in
/// https://github.com/substrait-io/substrait/blob/main/proto/substrait/type.proto.
// TODO(ingomueller): Add the other low-hanging fruits here.
def Substrait_AtomicTypes {
  list<Type> types = [
    SI1, // Boolean
    SI8, // I8
    SI16, // I16
    SI32, // I32
    SI64, // I64
    F32, // FP32
    F64, // FP64
  ];
}

/// Attributes of currently supported atomic types.
def Substrait_AtomicAttributes {
  list<Attr> attrs = [
    SI1Attr, // Boolean
    SI8Attr, // I8
    SI16Attr, // I16
    SI32Attr, // I32
    SI64Attr, // I64
    F32Attr, // FP32
    F64Attr, // FP64
  ];
}

/// Attribute of one of the currently supported atomic types.
def Substrait_AtomicAttribute : AnyAttrOf<Substrait_AtomicAttributes.attrs>;

/// One of the currently supported atomic types.
def Substrait_AtomicType : AnyTypeOf<Substrait_AtomicTypes.types>;

/// Any container type, i.e., structs, maps, lists, and nestings thereof.
def Substrait_ContainerType : NestedTupleOf<Substrait_AtomicTypes.types>;

/// One of the currently supported atomic or nested types.
def Substrait_FieldType : AnyTypeOf<[
  Substrait_AtomicType,
  Substrait_ContainerType
]>;

/// Placeholder for a proper relation type, the result of any `RelOpInterface`
/// op.
// TODO(ingomueller): Transform this into a proper relation type.
def Substrait_Relation : NestedTupleOf<Substrait_AtomicTypes.types>;

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITTYPES
