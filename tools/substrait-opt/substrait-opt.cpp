//===-- substrait-opt.cpp - Optimizer Driver for Substrait MLIR -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
// Reuse Registration.cpp as much as possible.
//
//===----------------------------------------------------------------------===//

#include "substrait-mlir/Dialect/Substrait/IR/Substrait.h"
#include "substrait-mlir/Dialect/Substrait/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

#include <string>

using namespace mlir;
using namespace mlir::substrait;

namespace {
void registerSubstraitDialects(DialectRegistry &registry) {
  registry.insert<mlir::substrait::SubstraitDialect>();
}
} // namespace

int main(int argc, char **argv) {
  static std::string executable =
      llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  llvm::sys::PrintStackTraceOnErrorSignal(executable);

  registerAllPasses();
  registerSubstraitPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerSubstraitDialects(registry);

  return failed(
      MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry));
}
