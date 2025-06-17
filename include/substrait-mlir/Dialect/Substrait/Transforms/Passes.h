//===- Passes.h - Substrait pass declarations -------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_MLIR_DIALECT_SUBSTRAIT_TRANSFORMS_PASSES_H_
#define SUBSTRAIT_MLIR_DIALECT_SUBSTRAIT_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class RewritePatternSet;
}

namespace mlir {
namespace substrait {

#define GEN_PASS_DECL
#include "substrait-mlir/Dialect/Substrait/Transforms/Passes.h.inc" // IWYU pragma: export

/// Create a pass to eliminate duplicate fields in `emit` ops.
std::unique_ptr<Pass> createEmitDeduplicationPass();

/// Add patterns that eliminate duplicate fields in `emit` ops.
void populateEmitDeduplicationPatterns(RewritePatternSet &patterns);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "substrait-mlir/Dialect/Substrait/Transforms/Passes.h.inc" // IWYU pragma: export

} // namespace substrait
} // namespace mlir

#endif // SUBSTRAIT_MLIR_DIALECT_SUBSTRAIT_TRANSFORMS_PASSES_H_
