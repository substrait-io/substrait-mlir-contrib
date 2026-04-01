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
include "substrait-mlir/Dialect/Substrait/IR/SubstraitInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

// Base class for Substrait dialect types.
class Substrait_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Substrait_Dialect, name, traits> {
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

def Substrait_DecimalType : Substrait_Type<"Decimal", "decimal"> {
  let summary = "Substrait decimal type";
  let description = [{
    This type represents a substrait decimal type.
  }];
  let parameters = (ins "uint32_t":$precision, "uint32_t":$scale);
  let assemblyFormat = "`<` $precision `,` $scale `>`";
  let genVerifyDecl = 1;
}

def Substrait_FixedBinaryType : Substrait_Type<"FixedBinary", "fixed_binary"> {
  let summary = "Substrait fixed-length binary type";
  let description = [{
    This type represents a substrait binary string of L bytes.
  }];
  let parameters = (ins "int32_t":$length);
  let assemblyFormat = [{ `<` $length `>` }];
}

def Substrait_FixedCharType : Substrait_Type<"FixedChar", "fixed_char"> {
  let summary = "Substrait fixed-length char type";
  let description = [{
    This type represents a substrait fixed-length unicode string of `L`
    characters.
  }];
  let parameters = (ins "int32_t":$length);
  let assemblyFormat = [{ `<` $length `>` }];
}

def Substrait_IntervalDaySecondType : Substrait_Type<"IntervalDaySecond", "interval_day_second"> {
  let summary = "Substrait interval day to second type";
  let description = [{
    This type represents a substrait interval day to second type.
  }];
}

def Substrait_IntervalYearMonthType : Substrait_Type<"IntervalYearMonth", "interval_year_month"> {
  let summary = "Substrait interval year to month type";
  let description = [{
    This type represents a substrait interval year to month type.
  }];
}

def Substrait_StringType : Substrait_Type<"String", "string"> {
  let summary = "Substrait string type";
  let description = [{
    This type represents a substrait string type.
  }];
}

def Substrait_TimeType : Substrait_Type<"Time", "time"> {
  let summary = "Substrait time type";
  let description = [{
    This type represents a substrait time type.
  }];
}

def Substrait_TimestampType : Substrait_Type<"Timestamp", "timestamp"> {
  let summary = "Substrait timezone-unaware timestamp type";
  let description = [{
    This type represents a substrait timezone-unaware timestamp type.
  }];
}

def Substrait_TimestampTzType : Substrait_Type<"TimestampTz", "timestamp_tz"> {
  let summary = "Substrait timezone-aware timestamp type";
  let description = [{
    This type represents a substrait timezone-aware timestamp type.
  }];
}

/// Currently supported atomic types. These correspond directly to the types in
def Substrait_UUIDType : Substrait_Type<"UUID", "uuid"> {
  let summary = "Substrait uuid type";
  let description = [{
    This type represents a substrait uuid (universally-unique identifier) type.
    This is a 16-byte binary value.
  }];
}

def Substrait_VarCharType : Substrait_Type<"VarChar", "var_char"> {
  let summary = "Substrait variable-length char type";
  let description = [{
    This type represents a substrait variable-length unicode string of at most L
    characters.
  }];
  let parameters = (ins "int32_t":$length);
  let assemblyFormat = [{ `<` $length `>` }];
}

def Substrait_StructType : Substrait_Type<"Struct", "struct"> {
  let summary = "Substrait struct type";
  let description = [{
    Represents a Substrait `Type.struct` — an ordered sequence of field types.
    The nullability of the struct value itself is indicated by wrapping this
    type in `NullableType` (e.g., `struct<i32, string?>?` is a nullable struct
    with a required `i32` field and a nullable `string` field). Individual
    field nullability is expressed the same way as for any expression type.
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type", "field types">:$fieldTypes
  );
  let hasCustomAssemblyFormat = 1;
}

def Substrait_NullableType : Substrait_Type<"Nullable", "nullable"> {
  let summary = "Nullable wrapper for a Substrait expression type";
  let description = [{
    Wraps any Substrait expression type to indicate that values of this type may
    be null. The absence of this wrapper implies `NULLABILITY_REQUIRED`.
    `NULLABILITY_UNSPECIFIED` is imported as `NULLABILITY_NULLABLE`. The type is
    printed using the `T?` sugar syntax via a custom printer/parser in
    `Substrait.cpp`.
  }];
  let parameters = (ins "::mlir::Type":$innerType);
  // Default assembly format: "!substrait.nullable<T>".
  // The pretty "T?" form is handled by the custom SubstraitType printer/parser.
  let assemblyFormat = "`<` $innerType `>`";
  let genVerifyDecl = 1;
}

/// Currently supported simple types, listed in order of the Substrait specification.
/// These correspond directly to the types in
/// https://github.com/substrait-io/substrait/blob/main/proto/substrait/type.proto.
def Substrait_SimpleTypes {
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
    Substrait_TimeType, // Time
    Substrait_IntervalYearMonthType, // IntervalYear
    Substrait_IntervalDaySecondType, // IntervalDay
    Substrait_UUIDType, // UUID
  ];
}

/// Currently supported parametrized types, listed in order of the Substrait
/// specification. These correspond to the "compound" types in
/// https://github.com/substrait-io/substrait/blob/main/proto/substrait/type.proto.
def Substrait_ParametrizedTypes {
  list<Type> types = [
    Substrait_FixedCharType, // FixedChar
    Substrait_VarCharType, // VarChar
    Substrait_FixedBinaryType, // FixedBinary
    Substrait_DecimalType, // Decimal
  ];
}

def Substrait_AnyType : Substrait_Type<"Any", "any"> {
  let summary = "type of a 'google.protobuf.Any' protobuf message";
  let description = [{
    This type represents the `type_url` fields of a `google.protobuf.Any`
    message. These messages consist of an opaque byte array and a string holding
    the URL identifying the type of what is contained in the byte array.
  }];
  let parameters = (ins "StringAttr":$type_url);
  let assemblyFormat = "`<` $type_url `>`";

}

/// Currently supported atomic types, made up of simple and parametrized types.
def Substrait_AtomicTypes {
  list<Type> types =
    Substrait_SimpleTypes.types#Substrait_ParametrizedTypes.types;
}

/// One of the currently supported atomic types.
def Substrait_AtomicType :
  AnyTypeOf<Substrait_SimpleTypes.types#Substrait_ParametrizedTypes.types>;

/// Any container type; currently only `StructType` (`MapType`/`ListType` TBD).
def Substrait_ContainerType : AnyTypeOf<[Substrait_StructType]>;

/// Currently supported expression types.
def Substrait_ExpressionTypes {
  list<Type> types =
  Substrait_AtomicTypes.types;
  // TODO: add container types here once we have them.
}

/// One of the currently supported expression types, i.e., types of values that
/// can be the result of an expression (as opposed to results of relational
/// operators).
def Substrait_ExpressionType : AnyTypeOf<[
  Substrait_AtomicType,
  Substrait_ContainerType,
  Substrait_NullableType,
]>;

//===----------------------------------------------------------------------===//
// RelationType
//===----------------------------------------------------------------------===//

def Substrait_Relation : Substrait_Type<"Relation", "relation"> {
  let summary = "Substrait relation, i.e., result of `RelOpInterface` ops";
  let description = [{
  }];
  let parameters = (ins Substrait_StructType:$structType);
  let builders = [
    TypeBuilder<(ins "TypeRange":$fieldTypes), [{
      return $_get($_ctxt, StructType::get($_ctxt,
          SmallVector<Type>(fieldTypes.begin(), fieldTypes.end())));
    }]>,
    TypeBuilder<(ins), [{
      return $_get($_ctxt, StructType::get($_ctxt, SmallVector<Type>{}));
    }]>,
  ];
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Get the field types of the relation.
    ArrayRef<Type> getFieldTypes() const {
      return getStructType().getFieldTypes();
    }

    /// Return the number of fields.
    size_t size() const { return getFieldTypes().size(); }

    /// Iterate over the field types.
    using iterator = ArrayRef<Type>::iterator;
    iterator begin() const { return getFieldTypes().begin(); }
    iterator end() const { return getFieldTypes().end(); }
  }];
}

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITTYPES
