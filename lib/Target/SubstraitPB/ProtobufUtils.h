//===-- ProtobufUtils.h - Utils for Substrait protobufs ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TARGET_SUBSTRAITPB_PROTOBUFUTILS_H // NOLINT: can't detect root dir
#define LIB_TARGET_SUBSTRAITPB_PROTOBUFUTILS_H // NOLINT: can't detect root dir

#include "mlir/IR/Location.h"

#include <type_traits>

namespace substrait::proto {
class RelCommon;
class Rel;
} // namespace substrait::proto

namespace mlir::substrait::protobuf_utils {

/// Extract the `RelCommon` message from any possible `rel_type` message of the
/// given `rel`. Reports errors using the given `loc`.
FailureOr<const ::substrait::proto::RelCommon *>
getCommon(const ::substrait::proto::Rel &rel, Location loc);

/// Extract the `RelCommon` message from any possible `rel_type` message of the
/// given `rel`. Reports errors using the given `loc`.
FailureOr<::substrait::proto::RelCommon *>
getMutableCommon(::substrait::proto::Rel *rel, Location loc);

/// SFINAE-based template that checks if the given (message) type has an field
/// called `advanced_extension`: the `value` member is `true` iff it has. This
/// is useful to deal with the two different names, `advanced_extension` and
/// `advanced_extensions`, that are used for the same thing across different
/// message types in the Substrait spec.
template <typename T>
class HasAdvancedExtensions {
  template <typename C>
  static std::true_type test(decltype(&C::advanced_extensions));
  template <typename C>
  static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/// Trait class for accessing the `advanced_extension` field. The default
/// instances is automatically used for message types that call this field
/// `advanced_extension`; the specialization below is automatically used for
/// message types that call it `advanced_extensions`.
template <typename T, typename = void>
struct AdvancedExtensionTrait {
  static auto hasAdvancedExtension(const T &message) {
    return message.has_advanced_extension();
  }
  static auto advancedExtension(const T &message) {
    return message.advanced_extension();
  }
  template <typename S>
  static auto setAllocatedAdvancedExtension(T &message,
                                            S &&advancedExtensions) {
    message.set_allocated_advanced_extension(
        std::forward<S>(advancedExtensions));
  }
};

template <typename T>
struct AdvancedExtensionTrait<
    T, std::enable_if_t<HasAdvancedExtensions<T>::value>> {
  static auto hasAdvancedExtension(const T &message) {
    return message.has_advanced_extensions();
  }
  static auto advancedExtension(const T &message) {
    return message.advanced_extensions();
  }
  template <typename S>
  static auto setAllocatedAdvancedExtension(T &message,
                                            S &&advancedExtensions) {
    message.set_allocated_advanced_extensions(
        std::forward<S>(advancedExtensions));
  }
};

} // namespace mlir::substrait::protobuf_utils

#endif // LIB_TARGET_SUBSTRAITPB_PROTOBUFUTILS_H
