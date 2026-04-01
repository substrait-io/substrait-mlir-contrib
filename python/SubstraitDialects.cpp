//===-- SubstraitDialects.cpp - Python module for Substrait MLIR-*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "substrait-mlir-c/Dialects.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"         // IWYU pragma: keep
#include "mlir/Bindings/Python/NanobindAdaptors.h" // IWYU pragma: keep
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

#define SERIALIZATION_FORMAT                                                   \
  "substrait_mlir._mlir_libs._substraitDialects.substrait.SerializationFormat"

NB_MODULE(_substraitDialects, mainModule) {
#ifndef NDEBUG
  static std::string executable =
      llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  llvm::sys::PrintStackTraceOnErrorSignal(executable);
#endif

  //===--------------------------------------------------------------------===//
  // Substrait dialect.
  //===--------------------------------------------------------------------===//
  auto substraitModule = mainModule.def_submodule("substrait");

  //
  // Dialect
  //

  substraitModule.def(
      "register_dialect",
      [](MlirContext context, bool doLoad) {
        MlirDialectHandle handle = mlirGetDialectHandle__substrait__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (doLoad)
          mlirDialectHandleLoadDialect(handle, context);
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true,
      nb::sig("def register_dialect("
              "context: Optional[substrait_mlir.ir.Context] = None, "
              "load: bool = True) -> None"),
      "Register and optionally load the dialect with the given context");

  //
  // Enums
  //

  nb::enum_<MlirSubstraitSerializationFormat>(
      substraitModule, "SerializationFormat", nb::is_arithmetic(),
      nb::is_flag())
      .value("text", MlirSubstraitTextFormat)
      .value("binary", MlirSubstraitBinaryFormat)
      .value("json", MlirSubstraitJsonFormat)
      .value("pretty_json", MlirSubstraitPrettyJsonFormat);

  //
  // Types
  //

  mlir_type_subclass(substraitModule, "AnyType", mlirTypeIsASubstraitAnyType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, std::string typeUrl, MlirContext context) {
            MlirStringRef urlRef{typeUrl.data(), typeUrl.size()};
            return cls(mlirSubstraitAnyTypeGet(context, urlRef));
          },
          nb::arg("cls"), nb::arg("type_url"),
          nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, type_url: str, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> AnyType"));

  mlir_type_subclass(substraitModule, "BinaryType",
                     mlirTypeIsASubstraitBinaryType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirSubstraitBinaryTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> BinaryType"));

  mlir_type_subclass(substraitModule, "DateType", mlirTypeIsASubstraitDateType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirSubstraitDateTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> DateType"));

  mlir_type_subclass(substraitModule, "DecimalType",
                     mlirTypeIsASubstraitDecimalType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, uint32_t precision, uint32_t scale,
             MlirContext context) {
            return cls(mlirSubstraitDecimalTypeGet(context, precision, scale));
          },
          nb::arg("cls"), nb::arg("precision"), nb::arg("scale"),
          nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, precision: int, scale: int, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> DecimalType"));

  mlir_type_subclass(substraitModule, "FixedBinaryType",
                     mlirTypeIsASubstraitFixedBinaryType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, int32_t length, MlirContext context) {
            return cls(mlirSubstraitFixedBinaryTypeGet(context, length));
          },
          nb::arg("cls"), nb::arg("length"),
          nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, length: int, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> FixedBinaryType"));

  mlir_type_subclass(substraitModule, "FixedCharType",
                     mlirTypeIsASubstraitFixedCharType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, int32_t length, MlirContext context) {
            return cls(mlirSubstraitFixedCharTypeGet(context, length));
          },
          nb::arg("cls"), nb::arg("length"),
          nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, length: int, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> FixedCharType"));

  mlir_type_subclass(substraitModule, "IntervalDaySecondType",
                     mlirTypeIsASubstraitIntervalDaySecondType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirSubstraitIntervalDaySecondTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> IntervalDaySecondType"));

  mlir_type_subclass(substraitModule, "IntervalYearMonthType",
                     mlirTypeIsASubstraitIntervalYearMonthType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirSubstraitIntervalYearMonthTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> IntervalYearMonthType"));

  mlir_type_subclass(substraitModule, "NullableType",
                     mlirTypeIsASubstraitNullableType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirType innerType, MlirContext context) {
            return cls(mlirSubstraitNullableTypeGet(context, innerType));
          },
          nb::arg("cls"), nb::arg("inner_type"),
          nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "inner_type: substrait_mlir.ir.Type, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> NullableType"));

  mlir_type_subclass(substraitModule, "RelationType",
                     mlirTypeIsASubstraitRelationType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, std::vector<MlirType> fieldTypes,
             MlirContext context) {
            return cls(mlirSubstraitRelationTypeGet(context, fieldTypes.size(),
                                                    fieldTypes.data()));
          },
          nb::arg("cls"), nb::arg("element_type"),
          nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "element_type: typing.Sequence[substrait_mlir.ir.Type], "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> RelationType"));

  mlir_type_subclass(substraitModule, "StringType",
                     mlirTypeIsASubstraitStringType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirSubstraitStringTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> StringType"));

  mlir_type_subclass(substraitModule, "StructType",
                     mlirTypeIsASubstraitStructType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, std::vector<MlirType> fieldTypes,
             MlirContext context) {
            return cls(mlirSubstraitStructTypeGet(context, fieldTypes.size(),
                                                  fieldTypes.data()));
          },
          nb::arg("cls"), nb::arg("field_types"),
          nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "field_types: typing.Sequence[substrait_mlir.ir.Type], "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> StructType"));

  mlir_type_subclass(substraitModule, "TimeType", mlirTypeIsASubstraitTimeType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirSubstraitTimeTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> TimeType"));

  mlir_type_subclass(substraitModule, "TimestampType",
                     mlirTypeIsASubstraitTimestampType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirSubstraitTimestampTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> TimestampType"));

  mlir_type_subclass(substraitModule, "TimestampTzType",
                     mlirTypeIsASubstraitTimestampTzType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirSubstraitTimestampTzTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> TimestampTzType"));

  mlir_type_subclass(substraitModule, "UUIDType", mlirTypeIsASubstraitUUIDType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) {
            return cls(mlirSubstraitUUIDTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> UUIDType"));

  mlir_type_subclass(substraitModule, "VarCharType",
                     mlirTypeIsASubstraitVarCharType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, int32_t length, MlirContext context) {
            return cls(mlirSubstraitVarCharTypeGet(context, length));
          },
          nb::arg("cls"), nb::arg("length"),
          nb::arg("context").none() = nb::none(),
          nb::sig("def get(cls: object, length: int, "
                  "context: typing.Optional[substrait_mlir.ir.Context] = None) "
                  "-> VarCharType"));

  //
  // Import
  //
  substraitModule.def(
      "from_protobuf",
      [&](const nb::bytes &input, MlirSubstraitSerializationFormat format,
          MlirContext context) {
        MlirStringRef mlirInput{reinterpret_cast<const char *>(input.data()),
                                input.size()};
        MlirModule module = mlirSubstraitImportPlan(context, mlirInput, format);
        if (mlirModuleIsNull(module))
          throw std::invalid_argument("Could not import Substrait plan");
        return module;
      },
      nb::arg("input"), nb::arg("format") = MlirSubstraitTextFormat,
      nb::arg("context") = nb::none(),
      nb::sig("def from_protobuf(input: bytes, "
              "format: typing.Optional[" SERIALIZATION_FORMAT
              "] = " SERIALIZATION_FORMAT ".text, "
              "context: typing.Optional[substrait_mlir.ir.Context] = None)"
              "-> substrait_mlir.ir.Module"),
      "Import the Substrait plan in the given serialization format");

  //
  // Export
  //

  substraitModule.def(
      "to_protobuf",
      [&](MlirOperation op, MlirSubstraitSerializationFormat format) {
        MlirAttribute attr = mlirSubstraitExportPlan(op, format);
        if (mlirAttributeIsNull(attr))
          throw std::invalid_argument("Could not export Substrait plan");
        MlirStringRef strRef = mlirStringAttrGetValue(attr);
        std::string_view sv(strRef.data, strRef.length);
        return nb::bytes(sv.data(), sv.size());
      },
      nb::arg("op"), nb::arg("format") = MlirSubstraitTextFormat,
      nb::sig("def to_protobuf("
              "op: substrait_mlir.ir.Operation | substrait_mlir.ir.OpView, "
              "format: typing.Optional[" SERIALIZATION_FORMAT
              "] = " SERIALIZATION_FORMAT ".text)"
              "-> bytes"),
      "Export the Substrait plan into the given format");
}
