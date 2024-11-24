//===-- Export.h - Export Substrait dialect to protobuf ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_MLIR_TARGET_SUBSTRAITPB_EXPORT_H
#define SUBSTRAIT_MLIR_TARGET_SUBSTRAITPB_EXPORT_H

#include "substrait-mlir/Target/SubstraitPB/Options.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class LogicalResult;
}
namespace mlir {
class Operation;
using LogicalResult = llvm::LogicalResult;

namespace substrait {

LogicalResult
translateSubstraitToProtobuf(Operation *op, llvm::raw_ostream &output,
                             substrait::ImportExportOptions options = {});

} // namespace substrait
} // namespace mlir

#endif // SUBSTRAIT_MLIR_TARGET_SUBSTRAITPB_EXPORT_H
