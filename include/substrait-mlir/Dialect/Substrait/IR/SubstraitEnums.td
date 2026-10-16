//===-- SubstraitEnums.td - Substrait enums definitions ----*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITENUMS
#define SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITENUMS

include "mlir/IR/EnumAttr.td"

/// Represents the `AggregationInvocation` protobuf enum.
//
/// The enum values correspond exactly to those in the `JoinRel.JoinType` enum,
/// i.e., conversion through integers is possible.
def AggregationInvocation
    : I32EnumAttr<"AggregationInvocation", "aggregate invocation type", [
        // clang-format off
        I32EnumAttrCase<"unspecified", 0>,
        I32EnumAttrCase<"all", 1>,
        I32EnumAttrCase<"distinct", 2>,
        // clang-format on
      ]> {
  let cppNamespace = "::mlir::substrait";
}

/// Represents the `AggregationPhase` protobuf enum.
//
/// The enum values correspond exactly to those in the `JoinRel.JoinType` enum,
/// i.e., conversion through integers is possible.
def AggregationPhase
    : I32EnumAttr<"AggregationPhase", "aggregate invocation phase", [
        // clang-format off
        I32EnumAttrCase<"unspecified", 0>,
        I32EnumAttrCase<"initial_to_intermediate", 1>,
        I32EnumAttrCase<"intermediate_to_intermediate", 2>,
        I32EnumAttrCase<"initial_to_result", 3>,
        I32EnumAttrCase<"intermediate_to_result", 4>,
        // clang-format on
      ]> {
  let cppNamespace = "::mlir::substrait";
}

/// Represents the `FailureBehavior` enum.
///
/// The enum values correspond exactly to those in the `Cast.FailureBehavior`
/// enum, i.e., conversion through integers is possible.
def FailureBehavior : I32EnumAttr<"FailureBehavior", "", [
    // clang-format off
    I32EnumAttrCase<"unspecified", 0>,
    I32EnumAttrCase<"return_null", 1>,
    I32EnumAttrCase<"throw_exception", 2>,
    // clang-format on
  ]> {
  let cppNamespace = "::mlir::substrait";
}

/// Represents the `JoinType` protobuf enum.
///
/// The enum values correspond exactly to those in the `JoinRel.JoinType` enum,
/// i.e., conversion through integers is possible.
def JoinType : I32EnumAttr<"JoinType", "", [
    // clang-format off
    I32EnumAttrCase<"unspecified", 0>,
    I32EnumAttrCase<"inner", 1>,
    I32EnumAttrCase<"outer", 2>,
    I32EnumAttrCase<"left", 3>,
    I32EnumAttrCase<"right", 4>,
    I32EnumAttrCase<"semi", 5>,
    I32EnumAttrCase<"anti", 6>,
    I32EnumAttrCase<"single", 7>,
    // clang-format on
  ]> {
  let cppNamespace = "::mlir::substrait";
}

/// Represents the `SetOp` protobuf enum.
///
/// The enum values correspond exactly to those in the `SetRel.SetOp` enum,
/// i.e., conversion through integers is possible.
def SetOpKind : I32EnumAttr<"SetOpKind", "", [
    // clang-format off
    I32EnumAttrCase<"unspecified", 0>,
    I32EnumAttrCase<"minus_primary", 1>,
    I32EnumAttrCase<"minus_multiset", 2>,
    I32EnumAttrCase<"intersection_primary", 3>,
    I32EnumAttrCase<"intersection_multiset", 4>,
    I32EnumAttrCase<"union_distinct", 5>,
    I32EnumAttrCase<"union_all", 6>,
    // clang-format on
  ]> {
  let cppNamespace = "::mlir::substrait";
}

/// Represents the `SortField.SortDirection` protobuf enum.
///
/// The enum values correspond exactly to those in the
/// `SortField.SortDirection` enum, i.e., conversion through integers is
/// possible.
def SortFieldComparisonType : I32EnumAttr<"SortFieldComparisonType", "", [
    // clang-format off
    I32EnumAttrCase<"UNSPECIFIED", 0>,
    I32EnumAttrCase<"ASC_NULLS_FIRST", 1>,
    I32EnumAttrCase<"ASC_NULLS_LAST", 2>,
    I32EnumAttrCase<"DESC_NULLS_FIRST", 3>,
    I32EnumAttrCase<"DESC_NULLS_LAST", 4>,
    I32EnumAttrCase<"CLUSTERED", 5>,
    // clang-format on
  ]> {
  let cppNamespace = "::mlir::substrait";
}

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITENUMS
