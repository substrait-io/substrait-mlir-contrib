# Generated with plus manual fixes:
#   pybind11-stubgen substrait_mlir._mlir_libs._substraitDialects -o python`
from __future__ import annotations
from typing import Optional
from substrait_mlir.ir import Context, Module, Operation
__all__ = ['from_binpb', 'from_json', 'from_textpb', 'register_dialect', 'to_binpb', 'to_json', 'to_textpb']
def from_binpb(input, context: Optional[Context] = None) -> Module:
    """
    Import a Substrait plan in the binary protobuf format
    """
def from_json(input, context: Optional[Context] = None) -> Module:
    """
    Import a Substrait plan in the JSON format
    """
def from_textpb(input, context: Optional[Context] = None) -> Module:
    """
    Import a Substrait plan in the textual protobuf format
    """
def register_dialect(context: Optional[Context] = None, load: bool = True) -> None:
    ...
def to_binpb(op: Operation) -> str:
    """
    Export a Substrait plan into the binary protobuf format
    """
def to_json(op: Operation, pretty: bool = False) -> str:
    """
    Export a Substrait plan into the JSON format
    """
def to_textpb(op: Operation) -> str:
    """
    Export a Substrait plan into the textual protobuf format
    """
