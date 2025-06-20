//===-- substrait-lsp-server.cpp - Substrait MLIR LSP server ----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a substrait-specific MLIR LSP Language server. This
/// extends the as-you-type diagnostics in VS Code to dialects defined in this
/// repository. Implementation essentially as explained here:
/// https://mlir.llvm.org/docs/Tools/MLIRLSP/.
///
//===----------------------------------------------------------------------===//

#include "substrait-mlir/Dialect/Substrait/IR/Substrait.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

#include <string>

using namespace mlir;

namespace {
void registerSubstraitDialects(DialectRegistry &registry) {
  registry.insert<mlir::substrait::SubstraitDialect>();
}
} // namespace

int main(int argc, char **argv) {
#ifndef NDEBUG
  static std::string executable =
      llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  llvm::sys::PrintStackTraceOnErrorSignal(executable);
#endif

  registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerSubstraitDialects(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
