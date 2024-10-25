//===-- Import.h - Import protobuf to Substrait dialect ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_MLIR_TARGET_SUBSTRAITPB_IMPORT_H
#define SUBSTRAIT_MLIR_TARGET_SUBSTRAITPB_IMPORT_H

#include "substrait-mlir/Target/SubstraitPB/Options.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

class MLIRContext;
class ModuleOp;
template <typename T>
class OwningOpRef;

namespace substrait {

OwningOpRef<ModuleOp>
translateProtobufToSubstrait(llvm::StringRef input, MLIRContext *context,
                             substrait::ImportExportOptions options = {});

} // namespace substrait
} // namespace mlir

#endif // SUBSTRAIT_MLIR_TARGET_SUBSTRAITPB_IMPORT_H
