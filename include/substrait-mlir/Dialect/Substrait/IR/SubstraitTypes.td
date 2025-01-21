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
include "mlir/IR/BuiltinAttributeInterfaces.td"

// Base class for Substrait dialect types.
class Substrait_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Substrait_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Base class for Substrait dialect attribute types.
class Substrait_Attr<string name, string typeMnemonic, list<Trait> traits = []>
    : AttrDef<Substrait_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Substrait_BinaryType : Substrait_Type<"Binary", "binary"> {
  let summary = "Substrait binary type";
  let description = [{
    This type represents a substrait binary type.
  }];
}

def Substrait_DateType : Substrait_Type<"Date", "date"> {
  let summary = "Substrait date type";
  let description = [{
    This type represents a substrait date type. 
  }];
}

def Substrait_DateAttr : Substrait_Attr<"Date", "date",
  [TypedAttrInterface]> {
  let summary = "Substrait date type";
  let description = [{
    This type represents a substrait date attribute type.
  }];  
  let parameters = (ins "int32_t":$value);
  let assemblyFormat = [{ `<` $value `>` }];
  let extraClassDeclaration = [{ 
    ::mlir::Type getType() const {
      return DateType::get(getContext());
    }
  }];
}

def Substrait_StringType : Substrait_Type<"String", "string"> {
  let summary = "Substrait string type";
  let description = [{
    This type represents a substrait string type.
  }];
}

def Substrait_TimestampType : Substrait_Type<"Timestamp", "timestamp"> {
  let summary = "Substrait timezone-unaware timestamp type";
  let description = [{
    This type represents a substrait timezone-unaware timestamp type. 
  }];
}

def Substrait_TimestampAttr : Substrait_Attr<"Timestamp", "timestamp",
  [TypedAttrInterface]> {
  let summary = "Substrait timezone-unaware timestamp type";
  let description = [{
    This type represents a substrait timezone-unaware timestamp attribute type.
  }];  
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = [{ `<` $value `` `us` `>` }];
  let extraClassDeclaration = [{ 
    ::mlir::Type getType() const {
      return TimestampType::get(getContext());
    }
  }];
}

def Substrait_TimestampTzType : Substrait_Type<"TimestampTz", "timestamp_tz"> {
  let summary = "Substrait timezone-aware timestamp type";
  let description = [{
    This type represents a substrait timezone-aware timestamp type. 
  }];
}

def Substrait_TimestampTzAttr : Substrait_Attr<"TimestampTz", "timestamp_tz",
  [TypedAttrInterface]> {
  let summary = "Substrait timezone-aware timestamp type";
  let description = [{
    This type represents a substrait timezone-aware timestamp attribute type.
  }];  
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = [{ `<` $value `` `us` `>` }];
  let extraClassDeclaration = [{ 
    ::mlir::Type getType() const {
      return TimestampTzType::get(getContext());
    }
  }];
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
    Substrait_StringType, // String
    Substrait_BinaryType, // Binary
    Substrait_TimestampType, // Timestamp
    Substrait_TimestampTzType, // TimestampTZ
    Substrait_DateType, // Date
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
    TypedStrAttr<Substrait_StringType>, // String
    TypedStrAttr<Substrait_BinaryType>, // Binary
    Substrait_TimestampAttr, // Timestamp
    Substrait_TimestampTzAttr, // TimestampTZ
    Substrait_DateAttr, // Date
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
