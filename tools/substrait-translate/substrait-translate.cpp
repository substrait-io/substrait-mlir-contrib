//===-- substrait-translate.cpp - mlir-translate for Substrait --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// `mlir-stranslate` with translations to and from Substrait MLIR dialects.
//===----------------------------------------------------------------------===//

#include "substrait-mlir/Dialect/Substrait/IR/Substrait.h"
#include "substrait-mlir/Target/SubstraitPB/Export.h"
#include "substrait-mlir/Target/SubstraitPB/Import.h"
#include "substrait-mlir/Target/SubstraitPB/Options.h"

#include "mlir/IR/BuiltinOps.h" // IWYU pragma: keep
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::substrait;

namespace {
void registerSubstraitDialects(DialectRegistry &registry) {
  registry.insert<mlir::substrait::SubstraitDialect>();
}

llvm::cl::opt<SerdeFormat> substraitProtobufFormat(
    "substrait-protobuf-format", llvm::cl::ValueRequired,
    llvm::cl::desc(
        "Serialization format used when translating Substrait plans."),
    llvm::cl::values(
        clEnumValN(SerdeFormat::kText, "text", "human-readable text format"),
        clEnumValN(SerdeFormat::kBinary, "binary", "binary wire format"),
        clEnumValN(SerdeFormat::kJson, "json", "compact JSON format"),
        clEnumValN(SerdeFormat::kPrettyJson, "pretty-json",
                   "JSON format with new lines")),
    llvm::cl::init(SerdeFormat::kText));

void registerSubstraitToProtobufTranslation() {
  TranslateFromMLIRRegistration registration(
      "substrait-to-protobuf", "translate from Substrait MLIR to protobuf",
      [&](mlir::Operation *op, llvm::raw_ostream &output) {
        ImportExportOptions options;
        options.serdeFormat = substraitProtobufFormat.getValue();
        return translateSubstraitToProtobuf(op, output, options);
      },
      registerSubstraitDialects);
}

void registerProtobufToSubstraitPlanTranslation() {
  // Two aliases: `-protobuf-to-substrait` for conciseness and
  // `-protobuf-to-substrait-plan` for symmetry with the other translations.
  for (StringRef suffix : {"", "-plan"}) {
    TranslateToMLIRRegistration registration(
        ("protobuf-to-substrait" + suffix).str(),
        "translate a protobuf 'Plan' to Substrait MLIR",
        [&](llvm::StringRef input, mlir::MLIRContext *context) {
          ImportExportOptions options;
          options.serdeFormat = substraitProtobufFormat.getValue();
          return translateProtobufToSubstraitPlan(input, context, options);
        },
        registerSubstraitDialects);
  }
}

void registerProtobufToSubstraitPlanVersionTranslation() {
  TranslateToMLIRRegistration registration(
      "protobuf-to-substrait-plan-version",
      "translate a protobuf 'PlanVersion' to Substrait MLIR",
      [&](llvm::StringRef input, mlir::MLIRContext *context) {
        ImportExportOptions options;
        options.serdeFormat = substraitProtobufFormat.getValue();
        return translateProtobufToSubstraitPlanVersion(input, context, options);
      },
      registerSubstraitDialects);
}
} // namespace

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  registerSubstraitToProtobufTranslation();
  registerProtobufToSubstraitPlanTranslation();
  registerProtobufToSubstraitPlanVersionTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
