//===-- SubstraitAttrs.td - Substrait dialect attributes ---*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITATTRS
#define SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITATTRS

include "substrait-mlir/Dialect/Substrait/IR/SubstraitDialect.td"
include "substrait-mlir/Dialect/Substrait/IR/SubstraitTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

// Base class for Substrait dialect attribute types.
class Substrait_Attr<string name, string typeMnemonic, list<Trait> traits = []>
    : AttrDef<Substrait_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Substrait_AdvancedExtensionAttr
    : Substrait_Attr<"AdvancedExtension", "advanced_extension"> {
  let summary = "Represents the `AdvancedExtenssion` message of Substrait";
  let parameters = (ins
    OptionalParameter<"StringAttr">:$optimization,
    OptionalParameter<"StringAttr">:$enhancement
  );
  let assemblyFormat = [{
    ( `optimization` `=` $optimization^ )?
    ( `enhancement`  `=` $enhancement^  )?
  }];
  let genVerifyDecl = 1;
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

def Substrait_TimeAttr : Substrait_Attr<"Time", "time",
    [TypedAttrInterface]> {
  let summary = "Substrait time type";
  let description = [{
    This type represents a substrait time attribute type.
  }];
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = [{ `<` $value `` `us` `>` }];
  let extraClassDeclaration = [{
    ::mlir::Type getType() const {
      return TimeType::get(getContext());
    }
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

/// Attributes of currently supported atomic types, listed in order of substrait
/// specification.
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
    Substrait_TimeAttr, // Time
  ];
}

/// Attribute of one of the currently supported atomic types.
def Substrait_AtomicAttribute : AnyAttrOf<Substrait_AtomicAttributes.attrs>;

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITATTRS
