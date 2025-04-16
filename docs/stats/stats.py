# ===-- stats.py - Generate coverage stats for the Substrait dialect ------=== #
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #

import argparse
import glob
import itertools
import sys

import google.protobuf as pb
import google.protobuf.descriptor
import google.protobuf.message
import google.protobuf.text_format
import pandas as pd
import substrait.gen.proto as sp
import substrait.gen.proto.capabilities_pb2
import substrait.gen.proto.extended_expression_pb2
import substrait.gen.proto.function_pb2
import substrait.gen.proto.plan_pb2


def read_messages_from_file(file_path, message_type):
  """
  Reads all messages of the given message type from the file at the given path.

  Args:
    file_path: The path to the file, which contains messages separated with a
      separator.
    message_type: The message type to read.

  Yields:
    One parsed protobuf message of the given type per file split.
  """
  with open(file_path, "r") as f:
    for split in f.read().split("# -----"):
      message = message_type()
      yield pb.text_format.Parse(split, message)


def get_used_fields(message):
  """
  Yiels all (potentially nested) fields of the given message that are set.

  Args:
    message: Message from which to extract the fields.

  Yields:
    Field descriptors for every (potentially nested) field that is set in the
    message.
  """
  if not isinstance(message, pb.message.Message):
    return

  for field_descriptor in message.DESCRIPTOR.fields:
    if field_descriptor.label == pb.descriptor.FieldDescriptor.LABEL_REPEATED:
      # Handle repeated fields
      nested_msgs = getattr(message, field_descriptor.name)
      if len(nested_msgs):
        yield field_descriptor
        for nested_msg in nested_msgs:
          yield from get_used_fields(nested_msg)
    elif field_descriptor.type == pb.descriptor.FieldDescriptor.TYPE_MESSAGE:
      # Handle message fields
      if message.HasField(field_descriptor.name):
        yield field_descriptor
        nested_msg = getattr(message, field_descriptor.name)
        yield from get_used_fields(nested_msg)
    else:
      # Handle scalar fields
      value = getattr(message, field_descriptor.name)
      if value != field_descriptor.default_value:
        yield field_descriptor


def full_message_type_name(descriptor):
  """Computes the full name of the given message descriptor."""
  container = descriptor.containing_type
  if container is None:
    return descriptor.name
  return full_message_type_name(container) + "." + descriptor.name


def get_field_container(descriptor):
  """Gets the container of the given descriptor."""
  return descriptor.containing_type or descriptor.containing_oneof


def full_field_name(descriptor):
  """Computes the full name of the given field descriptor."""
  container = get_field_container(descriptor)
  return descriptor.file.name + ":" + full_message_type_name(
      container) + "." + descriptor.name


def collect_all_message_types(message_descriptor, result=None):
  """
  Collects all unique message types within a protobuf message and its nested
  types.

  Args:
    message_descriptor: The descriptor of the protobuf message type.
    visited: A set to keep track of visited message types.

  Returns:
    A set of message descriptors for all unique message types.
  """
  if result is None:
    result = set()
  if message_descriptor not in result:
    result.add(message_descriptor)
    for field in message_descriptor.fields:
      if field.type == pb.descriptor.FieldDescriptor.TYPE_MESSAGE:
        collect_all_message_types(field.message_type, result)
    for cls in message_descriptor.nested_types:
      collect_all_message_types(cls, result)
    for other_message in message_descriptor.file.message_types_by_name.values():
      collect_all_message_types(other_message, result)
  return result


def collect_all_substrait_message_types():
  """Collects all message types defined in the Substrait specification."""
  all_message_types = set()
  for msg_type in [
      sp.capabilities_pb2.Capabilities,
      sp.extended_expression_pb2.ExtendedExpression, sp.plan_pb2.Plan,
      sp.function_pb2.FunctionSignature
  ]:
    collect_all_message_types(msg_type.DESCRIPTOR, all_message_types)
  return all_message_types


def get_field_type(field):
  """Gets type name of the given field descriptor. This is the full message type
  name if the field is an enum or a message or a type name such as `double` for
  built-in types."""
  if field.type == pb.descriptor.FieldDescriptor.TYPE_ENUM:
    return field.enum_type.full_name

  if field.type == pb.descriptor.FieldDescriptor.TYPE_MESSAGE:
    return field.message_type.full_name

  # Map cpp_type to type name
  cpp_type_name_map = {
      1: "double",
      2: "float",
      3: "int64",
      4: "uint64",
      5: "int32",
      6: "uint64",
      7: "int32",
      8: "bool",
      9: "string",
      10: "message",
      11: "bytes",
      12: "uint32",
      13: "enum",
      14: "sfixed32",
      15: "sfixed64",
      16: "sint32",
      17: "sint64",
  }
  return cpp_type_name_map.get(field.cpp_type, str(field.cpp_type))


def compute_used_fields(path_or_glob):
  """
  Computes all fields used in the messages in the given path or glob.

  Args:
    path_or_glob: The path or glob to the message files.

  Returns:
    The set descriptors of all fields used in the messages.
  """
  paths = glob.glob(path_or_glob)
  # TODO: should we also read `PlanVersion` and other top-level messages?
  messages = (m for path in paths
              for m in read_messages_from_file(path, sp.plan_pb2.Plan))
  return set(
      itertools.chain.from_iterable(
          get_used_fields(message) for message in messages))


def compute_covered_fields(all_fields, used_fields):
  """
  Computes which of the fields given in `all_fields` are covered in the fields
  given in `used_fields`.

  Args:
    all_fields: An iterable providing all fields.
    used_fields: An iterable providing the fields that are used.

  Returns:
    A dataframe with coverage stats multi-indexd on the message hierarchy.
  """

  # Load into dataframes.
  df_all = pd.DataFrame(data=all_fields, columns=["field"])
  df_used = pd.DataFrame(data=used_fields, columns=["field"])

  # Add `full_name` columns
  df_all["full_name"] = df_all["field"].apply(full_field_name)
  df_used["full_name"] = df_used["field"].apply(full_field_name)

  # Mark used field as `covered`.
  df_used["covered"] = 1
  df_used = df_used.drop("field", axis="columns")

  # Left join on `full_name`, mark coverage of unused fields as `0`.
  df = pd.merge(df_all, df_used, on="full_name", how="left")
  df.covered = df.covered.fillna(0).astype(int)

  # Compute additional metadata for each field.
  df["file"] = df["field"].apply(lambda x: x.file.name)
  df["type"] = df["field"].apply(get_field_type)
  df["name"] = df["field"].apply(lambda x: x.name)
  df["message"] = df["field"].apply(get_field_container)
  df["message_name"] = df["message"].apply(full_message_type_name)

  # Remove messages that are not from the spec.
  df = df[df.file.str.startswith("proto/")]

  # Create a multi-index for the message hierarchy.
  message_names = df["message_name"].apply(lambda x: x.split("."))
  depth = message_names.apply(len).max()
  tuples = message_names.apply(lambda x: x + [""] * (depth - len(x))).apply(
      tuple)
  msg_level_names = list(f"msg_lvl{i}" for i in range(depth))
  idx = pd.MultiIndex.from_tuples(tuples, names=msg_level_names)
  df = df.set_index(idx)

  # Remove temporary columns.
  df = df.drop(columns=["field", "message", "full_name"], axis="columns")

  return df


MARKDOWN_BODY = """# Coverage Stats

This file shows how much of the Substrait specification this project currently
covers. The coverage is computed by comparing the number of fields in the
specification (i.e., the `.proto` files) with the number of fields occurring
in the unit tests in this repository. This is approximate in several ways, for
example, does it not account for how many enum values are covered, but it is
reasonably easy to compute and should still provide an indication for progress.

## Coverage per `.proto` file

The following table shows the coverage per `.proto` file, i.e., for each of the
`.proto` files in the main Substrait repository, it counts all fields at all
nesting level as the `total` and compares that with the number of fields that
occur in the unit tests.

{file_stats}

## Coverage per top-level message type

The following table shows a drill-down into the top-level message types of each
file, i.e., for each file, for each message type defined at the top level of
that file, the number of `total` fields is counted and compare to the number of
fields that are covered in the unit tests.

{toplevel_stats}

## Coverage per `Type` and `Expression` submessage types

The following table shows a drill-down into the `Type` and `Expression`
submessages. Those are some of the most interesting parts of the specification
but not defined in top-level message types, so they are not visible in the
previous table.

{type_expr_stats}"""


def group_and_compute_stats(df, groupby):
  """
  Groups the given dataframe by the given columns and computes stats at that
  granularity.
  """
  df_g = df.groupby(groupby).aggregate({"covered": ["sum", "count"]})
  df_g = df_g.reset_index()
  df_g["fraction"] = df_g["covered"]["sum"] / df_g["covered"]["count"]
  df_g = df_g.sort_values(["fraction"] + groupby,
                          ascending=[False] + ([True] * len(groupby)))
  df_g["fraction"] = (df_g["fraction"] * 100).astype(int).astype(str) + " %"
  df_g.columns = groupby + ["covered (number)", "total", "covered (fraction)"]
  return df_g


def compute_markdown_doc(df):
  # Compute stats at file granularity.
  file_stats = group_and_compute_stats(df, ["file"])

  # Compute stats at top-level message granularity.
  toplevel_stats = group_and_compute_stats(df, ["file", "msg_lvl0"])
  toplevel_stats.rename(columns={"msg_lvl0": "top-level message type"},
                        inplace=True)

  # Compute stats at second-level message granularity for types and exprs.
  df_type_expr = df[((df["msg_lvl0"] == 'Type') |
                     (df["msg_lvl0"] == 'Expression')) & (df["msg_lvl1"] != '')]
  type_expr_stats = group_and_compute_stats(df_type_expr,
                                            ["file", "msg_lvl0", "msg_lvl1"])
  toplevel_stats.rename(columns={
      "msg_lvl0": "top-level message type",
      "msg_lvl1": "submessage type",
  },
                        inplace=True)

  def to_markdown(df):
    return df.to_markdown(tablefmt="github", index=False)

  # Convert dataframes to markdown tables and put them all together.
  return MARKDOWN_BODY.format(file_stats=to_markdown(file_stats),
                              toplevel_stats=to_markdown(toplevel_stats),
                              type_expr_stats=to_markdown(type_expr_stats))


def main():
  parser = argparse.ArgumentParser(prog='ProgramName',
                                   description='What the program does')
  parser.add_argument('-m',
                      '--messages',
                      required=True,
                      help='Path or GLOB to message files')
  parser.add_argument('-a',
                      '--action',
                      choices=[
                          'print-markdown', 'print-used-fields',
                          'print-spec-fields', 'print-all'
                      ],
                      default='print-markdown',
                      help='Action to perform')
  args = parser.parse_args()

  #
  # Step 1: Collect the fields used in the provided files.
  #

  used_fields = compute_used_fields(args.messages)

  if (args.action == 'print-used-fields'):
    print('-' * 80)
    print('Used fields:')
    print('-' * 80)
    for field in sorted(full_field_name(field) for field in used_fields):
      print('-', field)
    sys.exit(0)

  #
  # Step 2: Collect all unique message types from the spec.
  #

  all_message_types = collect_all_substrait_message_types()

  if (args.action == 'print-spec-fields'):
    print('-' * 80)
    print('All fields:')
    print('-' * 80)
    for msg_type in all_message_types:
      for field in msg_type.fields:
        print(f"- {full_field_name(field)} ({get_field_type(field)})")
    sys.exit(0)

  all_fields = (
      field for msg_type in all_message_types for field in msg_type.fields)

  #
  # Step 3: Compute coverage.
  #

  df = compute_covered_fields(all_fields, used_fields)

  if args.action == 'print-markdown':
    print(compute_markdown_doc(df.reset_index()))
    sys.exit(0)

  if args.action == 'print-all':
    for i in range(len(df.index.names) + 1):
      groupby = ["file"] + df.index.names[0:i]
      df_g = group_and_compute_stats(df, groupby)
      print(df_g.to_markdown(tablefmt="github", index=False))
      print()
    sys.exit(0)


if __name__ == "__main__":
  main()
