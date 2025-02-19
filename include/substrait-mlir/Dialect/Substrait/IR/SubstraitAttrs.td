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
class Substrait_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Substrait_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Base class for Substrait dialect attribute types that have a statically known
// value type.
class Substrait_StaticallyTypedAttr<string name, string attrMnemonic,
                                    string typeName, list<Trait> traits = []>
    : Substrait_Attr<
          name, attrMnemonic,
          traits#[DeclareAttrInterfaceMethods<TypeInferableAttrInterface>]> {
  let extraClassDeclaration = [{
    /// Implement TypeInferableAttrInterface.
    ::mlir::Type getType() {
      return ::mlir::substrait::}]#typeName#[{::get(getContext());
    }
  }];
}

/// `StringAttr` parameter that is the empty string by default.
def Substrait_DefaultEmptyStringParameter
    : DefaultValuedParameter<"StringAttr", [{mlir::StringAttr::get($_ctxt, "")}]>;

//===----------------------------------------------------------------------===//
// Substrait attributes
//===----------------------------------------------------------------------===//

/// Attribute used for `google.protobuf.Any` messages.
def Substrait_AnyAttr : TypedStrAttr<Substrait_AnyType>;

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

def Substrait_DateAttr
    : Substrait_StaticallyTypedAttr<"Date", "date", "DateType"> {
  let summary = "Substrait date type";
  let description = [{
    This type represents a substrait date attribute type.
  }];
  let parameters = (ins "int32_t":$value);
  let assemblyFormat = [{ `<` $value `>` }];
}

def Substrait_IntervalDaySecondAttr
    : Substrait_StaticallyTypedAttr<"IntervalDaySecond", "interval_day_second",
                                    "IntervalDaySecondType"> {
  let summary = "Substrait interval day to second type";
  let description = [{
    This type represents a substrait interval day to second attribute type. Note
    that this attribute does not attempt to canonicalize equivalent day-second
    intervals.
  }];
  let parameters = (ins "int32_t":$days, "int32_t":$seconds);
  let assemblyFormat = [{ `<` $days `` `d` $seconds `` `s` `>` }];
  let genVerifyDecl = 1;
}

def Substrait_IntervalYearMonthAttr
    : Substrait_StaticallyTypedAttr<"IntervalYearMonth", "interval_year_month",
                                    "IntervalYearMonthType"> {
  let summary = "Substrait interval year to month type";
  let description = [{
    This type represents a substrait interval year to month attribute type. Note
    that this attribute does not attempt to canonicalize equivalent year-month
    intervals.
  }];
  let parameters = (ins "int32_t":$years, "int32_t":$months);
  let assemblyFormat = [{ `<` $years `` `y` $months `` `m` `>` }];
  let genVerifyDecl = 1;
}

def Substrait_TimeAttr
    : Substrait_StaticallyTypedAttr<"Time", "time", "TimeType"> {
  let summary = "Substrait time type";
  let description = [{
    This type represents a substrait time attribute type.
  }];
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = [{ `<` $value `` `us` `>` }];
}

def Substrait_TimestampAttr
    : Substrait_StaticallyTypedAttr<"Timestamp", "timestamp", "TimestampType"> {
  let summary = "Substrait timezone-unaware timestamp type";
  let description = [{
    This type represents a substrait timezone-unaware timestamp attribute type.
  }];
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = [{ `<` $value `` `us` `>` }];
}

def Substrait_TimestampTzAttr
    : Substrait_StaticallyTypedAttr<"TimestampTz", "timestamp_tz",
                                    "TimestampTzType"> {
  let summary = "Substrait timezone-aware timestamp type";
  let description = [{
    This type represents a substrait timezone-aware timestamp attribute type.
  }];
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = [{ `<` $value `` `us` `>` }];
}

def Substrait_UUIDAttr
    : Substrait_StaticallyTypedAttr<"UUID", "uuid",
                                    "UUIDType"> {
  let summary = "Substrait uuid type";
  let description = [{
    This type represents a substrait uuid (universally-unique identifier)
    attribute type. This is a 16-byte binary value.
  }];
  let parameters = (ins "IntegerAttr":$value);
  //TODO: print the UUID in typical format
  let assemblyFormat = [{ `<` $value `>` }];
}

def Substrait_VersionAttr : Substrait_Attr<"Version", "version"> {
  let summary = "Substrait version";
  let description = [{
    Represents the `Version` message type.
  }];
  let parameters = (ins
    "uint32_t":$major_number,
    "uint32_t":$minor_number,
    "uint32_t":$patch_number,
    Substrait_DefaultEmptyStringParameter:$git_hash,
    Substrait_DefaultEmptyStringParameter:$producer

  );
  // TODO(ingomueller): make this even nicer with custom printer/parser
  let assemblyFormat = [{
    `` $major_number `` `:` `` $minor_number `` `:` `` $patch_number
    (`git_hash` $git_hash^)? (`producer` $producer^)?
  }];
  let builders = [
    AttrBuilder<(ins "uint32_t":$major, "uint32_t":$minor, "uint32_t":$patch), [{
      return $_get($_ctxt, major, minor, patch,
                   /*git_hash=*/StringAttr(),
                   /*producer=*/StringAttr());
    }]>,
    AttrBuilder<(ins "uint32_t":$major, "uint32_t":$minor, "uint32_t":$patch,
                    "::llvm::StringRef":$git_hash,
                    "::llvm::StringRef":$producer), [{
      auto gitHashAttr = ::mlir::StringAttr::get($_ctxt, git_hash);
      auto producerAttr = ::mlir::StringAttr::get($_ctxt, producer);
      return $_get($_ctxt, major, minor, patch, gitHashAttr, producerAttr);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Helpers and constraints
//===----------------------------------------------------------------------===//

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
    Substrait_IntervalYearMonthAttr, // IntervalYear
    Substrait_IntervalDaySecondAttr, // IntervalDay
    Substrait_UUIDAttr, // UUID
  ];
}

/// Attribute of one of the currently supported atomic types.
def Substrait_AtomicAttribute : AnyAttrOf<Substrait_AtomicAttributes.attrs>;

/// `ArrayAttr` of `ArrayAttr`s if `i64`s.
def I64ArrayArrayAttr : TypedArrayAttrBase<
    I64ArrayAttr, "64-bit integer array array attribute"
  >;

/// `ArrayAttr` of `ArrayAttr`s if `i64`s with at least one element.
def NonEmptyI64ArrayArrayAttr :
  ConfinedAttr<I64ArrayArrayAttr, [ArrayMinCount<1>]>;


#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITATTRS
