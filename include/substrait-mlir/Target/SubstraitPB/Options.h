//===-- Options.h - Options for import/and export of Substrait --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_MLIR_TARGET_SUBSTRAITPB_OPTIONS_H
#define SUBSTRAIT_MLIR_TARGET_SUBSTRAITPB_OPTIONS_H

#include "substrait-mlir/Dialect/Substrait/IR/Substrait.h"

namespace mlir {
namespace substrait {

struct ImportExportOptions {
  /// Specifies which serialization formats is used for serialization and
  /// deserialization to and from protobuf messages.
  SerializationFormat serializationFormat;
};

} // namespace substrait
} // namespace mlir

#endif // SUBSTRAIT_MLIR_TARGET_SUBSTRAITPB_OPTIONS_H
