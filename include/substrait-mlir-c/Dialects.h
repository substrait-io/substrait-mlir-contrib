//===-- Dialects.h - CAPI for dialects ----------------------------*- C -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_MLIR_C_DIALECTS_H
#define SUBSTRAIT_MLIR_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Substrait dialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Substrait, substrait);

/// Checks whether the given type is an `AnyType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitAnyType(MlirType type);

/// Gets the `AnyType` with the given `typeUrl` string.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitAnyTypeGet(MlirContext context, MlirStringRef typeUrl);

/// Checks whether the given type is a `BinaryType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitBinaryType(MlirType type);

/// Gets the `BinaryType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitBinaryTypeGet(MlirContext context);

/// Checks whether the given type is a `DateType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitDateType(MlirType type);

/// Gets the `DateType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitDateTypeGet(MlirContext context);

/// Checks whether the given type is a `DecimalType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitDecimalType(MlirType type);

/// Gets the `DecimalType` with the given `precision` and `scale`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitDecimalTypeGet(MlirContext context, uint32_t precision,
                                     uint32_t scale);

/// Checks whether the given type is a `FixedBinaryType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitFixedBinaryType(MlirType type);

/// Gets the `FixedBinaryType` with the given `length`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitFixedBinaryTypeGet(MlirContext context, int32_t length);

/// Checks whether the given type is a `FixedCharType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitFixedCharType(MlirType type);

/// Gets the `FixedCharType` with the given `length`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitFixedCharTypeGet(MlirContext context, int32_t length);

/// Checks whether the given type is an `IntervalDaySecondType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitIntervalDaySecondType(MlirType type);

/// Gets the `IntervalDaySecondType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitIntervalDaySecondTypeGet(MlirContext context);

/// Checks whether the given type is an `IntervalYearMonthType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitIntervalYearMonthType(MlirType type);

/// Gets the `IntervalYearMonthType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitIntervalYearMonthTypeGet(MlirContext context);

/// Checks whether the given type is a `NullableType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitNullableType(MlirType type);

/// Gets the `NullableType` wrapping the given `innerType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitNullableTypeGet(MlirContext context, MlirType innerType);

/// Checks whether the given type is a `RelationType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitRelationType(MlirType type);

/// Gets the `RelationType` with the given `fieldTypes`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitRelationTypeGet(MlirContext context, intptr_t numFields,
                                      MlirType *fieldTypes);

/// Checks whether the given type is a `StringType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitStringType(MlirType type);

/// Gets the `StringType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitStringTypeGet(MlirContext context);

/// Checks whether the given type is a `StructType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitStructType(MlirType type);

/// Gets the `StructType` with the given `fieldTypes`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitStructTypeGet(MlirContext context, intptr_t numFields,
                                    MlirType *fieldTypes);

/// Checks whether the given type is a `TimeType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitTimeType(MlirType type);

/// Gets the `TimeType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitTimeTypeGet(MlirContext context);

/// Checks whether the given type is a `TimestampType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitTimestampType(MlirType type);

/// Gets the `TimestampType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitTimestampTypeGet(MlirContext context);

/// Checks whether the given type is a `TimestampTzType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitTimestampTzType(MlirType type);

/// Gets the `TimestampTzType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitTimestampTzTypeGet(MlirContext context);

/// Checks whether the given type is a `UUIDType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitUUIDType(MlirType type);

/// Gets the `UUIDType`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitUUIDTypeGet(MlirContext context);

/// Checks whether the given type is a `VarCharType`.
MLIR_CAPI_EXPORTED
bool mlirTypeIsASubstraitVarCharType(MlirType type);

/// Gets the `VarCharType` with the given `length`.
MLIR_CAPI_EXPORTED
MlirType mlirSubstraitVarCharTypeGet(MlirContext context, int32_t length);

/// Serialization/deserialization format for exporting/importing Substrait
/// plans. The enum values correspond exactly to those in
/// `mlir::substrait::SerializationFormat`, i.e., conversion through integers
/// is possible.
typedef enum MlirSubstraitSerializationFormat {
  MlirSubstraitTextFormat,
  MlirSubstraitBinaryFormat,
  MlirSubstraitJsonFormat,
  MlirSubstraitPrettyJsonFormat
} MlirSubstraitSerdeFormat;

/// Imports a `Plan` message from `input`, which must be in the specified
/// serialization format. Returns a null module and emits diagnostics in case of
/// an error.
MLIR_CAPI_EXPORTED
MlirModule mlirSubstraitImportPlan(MlirContext context, MlirStringRef input,
                                   MlirSubstraitSerdeFormat format);

/// Exports the provided `substrait.plan` or `builtin.module` op to protobuf in
/// the specified serialization format stored in the value of a `StringAttr`.
/// Returns a null attribute and emits diagnostics in case of an error.
MLIR_CAPI_EXPORTED
MlirAttribute mlirSubstraitExportPlan(MlirOperation op,
                                      MlirSubstraitSerdeFormat format);

#ifdef __cplusplus
}
#endif

#endif // SUBSTRAIT_MLIR_C_DIALECTS_H
