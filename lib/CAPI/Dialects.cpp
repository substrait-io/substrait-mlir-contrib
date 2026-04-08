//===-- Dialects.cpp - CAPI for dialects ------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "substrait-mlir-c/Dialects.h"

#include "substrait-mlir/Dialect/Substrait/IR/Substrait.h"
#include "substrait-mlir/Target/SubstraitPB/Export.h"
#include "substrait-mlir/Target/SubstraitPB/Import.h"
#include "substrait-mlir/Target/SubstraitPB/Options.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::substrait;

//===----------------------------------------------------------------------===//
// Substrait dialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Substrait, substrait, SubstraitDialect)

bool mlirTypeIsASubstraitAnyType(MlirType type) {
  return mlir::isa<AnyType>(unwrap(type));
}

MlirType mlirSubstraitAnyTypeGet(MlirContext context, MlirStringRef typeUrl) {
  MLIRContext *ctx = unwrap(context);
  return wrap(AnyType::get(ctx, StringAttr::get(ctx, unwrap(typeUrl))));
}

bool mlirTypeIsASubstraitBinaryType(MlirType type) {
  return mlir::isa<BinaryType>(unwrap(type));
}

MlirType mlirSubstraitBinaryTypeGet(MlirContext context) {
  return wrap(BinaryType::get(unwrap(context)));
}

bool mlirTypeIsASubstraitDateType(MlirType type) {
  return mlir::isa<DateType>(unwrap(type));
}

MlirType mlirSubstraitDateTypeGet(MlirContext context) {
  return wrap(DateType::get(unwrap(context)));
}

bool mlirTypeIsASubstraitDecimalType(MlirType type) {
  return mlir::isa<DecimalType>(unwrap(type));
}

MlirType mlirSubstraitDecimalTypeGet(MlirContext context, uint32_t precision,
                                     uint32_t scale) {
  return wrap(DecimalType::get(unwrap(context), precision, scale));
}

bool mlirTypeIsASubstraitFixedBinaryType(MlirType type) {
  return mlir::isa<FixedBinaryType>(unwrap(type));
}

MlirType mlirSubstraitFixedBinaryTypeGet(MlirContext context, int32_t length) {
  return wrap(FixedBinaryType::get(unwrap(context), length));
}

bool mlirTypeIsASubstraitFixedCharType(MlirType type) {
  return mlir::isa<FixedCharType>(unwrap(type));
}

MlirType mlirSubstraitFixedCharTypeGet(MlirContext context, int32_t length) {
  return wrap(FixedCharType::get(unwrap(context), length));
}

bool mlirTypeIsASubstraitIntervalDaySecondType(MlirType type) {
  return mlir::isa<IntervalDaySecondType>(unwrap(type));
}

MlirType mlirSubstraitIntervalDaySecondTypeGet(MlirContext context) {
  return wrap(IntervalDaySecondType::get(unwrap(context)));
}

bool mlirTypeIsASubstraitIntervalYearMonthType(MlirType type) {
  return mlir::isa<IntervalYearMonthType>(unwrap(type));
}

MlirType mlirSubstraitIntervalYearMonthTypeGet(MlirContext context) {
  return wrap(IntervalYearMonthType::get(unwrap(context)));
}

bool mlirTypeIsASubstraitNullableType(MlirType type) {
  return mlir::isa<NullableType>(unwrap(type));
}

MlirType mlirSubstraitNullableTypeGet(MlirContext context, MlirType innerType) {
  return wrap(NullableType::get(unwrap(context), unwrap(innerType)));
}

bool mlirTypeIsASubstraitRelationType(MlirType type) {
  return mlir::isa<RelationType>(unwrap(type));
}

MlirType mlirSubstraitRelationTypeGet(MlirContext context, intptr_t numFields,
                                      MlirType *fieldTypes) {
  SmallVector<Type, 4> types;
  ArrayRef<Type> typesRef = unwrapList(numFields, fieldTypes, types);
  return wrap(RelationType::get(unwrap(context), typesRef));
}

bool mlirTypeIsASubstraitStringType(MlirType type) {
  return mlir::isa<StringType>(unwrap(type));
}

MlirType mlirSubstraitStringTypeGet(MlirContext context) {
  return wrap(StringType::get(unwrap(context)));
}

bool mlirTypeIsASubstraitStructType(MlirType type) {
  return mlir::isa<StructType>(unwrap(type));
}

MlirType mlirSubstraitStructTypeGet(MlirContext context, intptr_t numFields,
                                    MlirType *fieldTypes) {
  SmallVector<Type, 4> types;
  ArrayRef<Type> typesRef = unwrapList(numFields, fieldTypes, types);
  return wrap(StructType::get(unwrap(context), typesRef));
}

bool mlirTypeIsASubstraitTimeType(MlirType type) {
  return mlir::isa<TimeType>(unwrap(type));
}

MlirType mlirSubstraitTimeTypeGet(MlirContext context) {
  return wrap(TimeType::get(unwrap(context)));
}

bool mlirTypeIsASubstraitTimestampType(MlirType type) {
  return mlir::isa<TimestampType>(unwrap(type));
}

MlirType mlirSubstraitTimestampTypeGet(MlirContext context) {
  return wrap(TimestampType::get(unwrap(context)));
}

bool mlirTypeIsASubstraitTimestampTzType(MlirType type) {
  return mlir::isa<TimestampTzType>(unwrap(type));
}

MlirType mlirSubstraitTimestampTzTypeGet(MlirContext context) {
  return wrap(TimestampTzType::get(unwrap(context)));
}

bool mlirTypeIsASubstraitUUIDType(MlirType type) {
  return mlir::isa<UUIDType>(unwrap(type));
}

MlirType mlirSubstraitUUIDTypeGet(MlirContext context) {
  return wrap(UUIDType::get(unwrap(context)));
}

bool mlirTypeIsASubstraitVarCharType(MlirType type) {
  return mlir::isa<VarCharType>(unwrap(type));
}

MlirType mlirSubstraitVarCharTypeGet(MlirContext context, int32_t length) {
  return wrap(VarCharType::get(unwrap(context), length));
}

MlirModule mlirSubstraitImportPlan(MlirContext context, MlirStringRef input,
                                   MlirSubstraitSerdeFormat format) {
  ImportExportOptions options;
  options.serializationFormat = static_cast<SerializationFormat>(format);
  OwningOpRef<ModuleOp> owning =
      translateProtobufToSubstraitPlan(unwrap(input), unwrap(context), options);
  if (!owning)
    return MlirModule{nullptr};
  return MlirModule{owning.release().getOperation()};
}

MlirAttribute mlirSubstraitExportPlan(MlirOperation op,
                                      MlirSubstraitSerdeFormat format) {
  std::string str;
  llvm::raw_string_ostream stream(str);
  ImportExportOptions options;
  options.serializationFormat = static_cast<SerializationFormat>(format);
  if (failed(translateSubstraitToProtobuf(unwrap(op), stream, options)))
    return wrap(Attribute());
  MLIRContext *context = unwrap(op)->getContext();
  Attribute attr = StringAttr::get(context, str);
  return wrap(attr);
}
