//===-- ProtobufUtils.cpp - Utils for Substrait protobufs -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtobufUtils.h"
#include "mlir/IR/Diagnostics.h"

#include <substrait/proto/algebra.pb.h>

using namespace mlir;
using namespace ::substrait;
using namespace ::substrait::proto;

namespace pb = google::protobuf;

namespace mlir::substrait::protobuf_utils {

template <typename RelType>
static const RelCommon *getCommon(const RelType &rel) {
  return &rel.common();
}

FailureOr<const RelCommon *> getCommon(const Rel &rel, Location loc) {
  Rel::RelTypeCase relType = rel.rel_type_case();
  switch (relType) {
  case Rel::RelTypeCase::kAggregate:
    return getCommon(rel.aggregate());
  case Rel::RelTypeCase::kCross:
    return getCommon(rel.cross());
  case Rel::RelTypeCase::kFetch:
    return getCommon(rel.fetch());
  case Rel::RelTypeCase::kFilter:
    return getCommon(rel.filter());
  case Rel::RelTypeCase::kJoin:
    return getCommon(rel.join());
  case Rel::RelTypeCase::kProject:
    return getCommon(rel.project());
  case Rel::RelTypeCase::kRead:
    return getCommon(rel.read());
  case Rel::RelTypeCase::kSet:
    return getCommon(rel.set());
  default:
    const pb::FieldDescriptor *desc =
        Rel::GetDescriptor()->FindFieldByNumber(relType);
    return emitError(loc) << Twine("unsupported Rel type: ") + desc->name();
  }
}

template <typename RelType>
static RelCommon *getMutableCommon(RelType *rel) {
  return rel->mutable_common();
}

FailureOr<RelCommon *> getMutableCommon(Rel *rel, Location loc) {
  Rel::RelTypeCase relType = rel->rel_type_case();
  switch (relType) {
  case Rel::RelTypeCase::kAggregate:
    return getMutableCommon(rel->mutable_aggregate());
  case Rel::RelTypeCase::kCross:
    return getMutableCommon(rel->mutable_cross());
  case Rel::RelTypeCase::kFetch:
    return getMutableCommon((rel->mutable_fetch()));
  case Rel::RelTypeCase::kFilter:
    return getMutableCommon(rel->mutable_filter());
  case Rel::RelTypeCase::kJoin:
    return getMutableCommon(rel->mutable_join());
  case Rel::RelTypeCase::kProject:
    return getMutableCommon(rel->mutable_project());
  case Rel::RelTypeCase::kRead:
    return getMutableCommon(rel->mutable_read());
  case Rel::RelTypeCase::kSet:
    return getMutableCommon(rel->mutable_set());
  default:
    const pb::FieldDescriptor *desc =
        Rel::GetDescriptor()->FindFieldByNumber(relType);
    return emitError(loc) << Twine("unsupported Rel type: ") + desc->name();
  }
}

} // namespace mlir::substrait::protobuf_utils
