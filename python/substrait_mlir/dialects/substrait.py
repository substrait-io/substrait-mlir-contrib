# ===-- substrait.py - Imports and mixins for Substrait -------------------=== #
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #

from typing import Optional, Sequence

from ._substrait_ops_gen import *
from ._substrait_ops_gen import _Dialect
from .._mlir_libs._substraitDialects.substrait import *

try:
  from .. import ir
  from ._ods_common import _cext as _ods_cext
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e


def from_textpb(input: str, context: Optional[ir.Context] = None) -> ir.Module:
  """Import the Substrait plan from the textual protobuf format"""
  return from_protobuf(input.encode(), SerializationFormat.text, context)


def from_binpb(input: bytes, context: Optional[ir.Context] = None) -> ir.Module:
  """Import the Substrait plan from the binary protobuf format"""
  return from_protobuf(input, SerializationFormat.binary, context)


def from_json(input: str, context: Optional[ir.Context] = None) -> ir.Module:
  """Import the Substrait plan from the JSON protobuf format"""
  return from_protobuf(input.encode(), SerializationFormat.json, context)


def to_textpb(op: ir.Operation | ir.OpView) -> str:
  """Export the Substrait plan into the textual protobuf format"""
  return to_protobuf(op, SerializationFormat.text).decode()


def to_binpb(op: ir.Operation | ir.OpView) -> bytes:
  """Export the Substrait plan into the binary protobuf format"""
  return to_protobuf(op, SerializationFormat.binary)


def to_json(op: ir.Operation | ir.OpView, pretty: bool = False) -> str:
  """Export the Substrait plan into the JSON protobuf format"""
  if pretty:
    return to_protobuf(op, SerializationFormat.pretty_json).decode()
  return to_protobuf(op, SerializationFormat.json).decode()


@_ods_cext.register_operation(_Dialect, replace=True)
class PlanOp(PlanOp):

  def __init__(self, *args, version: Optional[Sequence[int]] = None, **kwargs):
    if version is not None:
      major, minor, patch = version
      for part in ["major", "minor", "patch"]:
        if (part + "_number") in kwargs:
          raise ValueError(
              "'version' and '(major|minor|patch)_number' are mutually exclusive"
          )
      args = (major, minor, patch) + args
    super().__init__(*args, **kwargs)
    self.regions[0].blocks.append()

  @property
  def body(self) -> ir.Block:
    return self.regions[0].blocks[0]

  def to_json(self, pretty: bool = False) -> str:
    return to_json(self, pretty)

  def to_binpb(self) -> bytes:
    return to_binpb(self)

  def to_textpb(self) -> str:
    return to_textpb(self)


@_ods_cext.register_operation(_Dialect, replace=True)
class PlanRelOp(PlanRelOp):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.regions[0].blocks.append()

  @property
  def body(self) -> ir.Block:
    return self.regions[0].blocks[0]
