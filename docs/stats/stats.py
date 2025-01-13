import argparse
import glob
import itertools

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

parser = argparse.ArgumentParser(prog='ProgramName',
                                 description='What the program does')
parser.add_argument('-m',
                    '--messages',
                    required=True,
                    help='Path or GLOB to message files')

args = parser.parse_args()


def read_messages_from_file(file_path, message_type):
  with open(file_path, "r") as f:
    for split in f.read().split("# -----"):
      message = message_type()
      yield pb.text_format.Parse(split, message)


def get_used_fields(message):
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
  container = descriptor.containing_type
  if container is None:
    return descriptor.name
  return full_message_type_name(container) + "." + descriptor.name


def get_field_container(descriptor):
  return descriptor.containing_type or descriptor.containing_oneof


def full_field_name(descriptor):
  container = get_field_container(descriptor)
  return descriptor.file.name + ":" + full_message_type_name(
      container) + "." + descriptor.name


paths = glob.glob(args.messages)
messages = (m for path in paths
            for m in read_messages_from_file(path, sp.plan_pb2.Plan))
used_fields = set(
    itertools.chain.from_iterable(
        get_used_fields(message) for message in messages))
used_filed_names = (full_field_name(field) for field in used_fields)

# print('-' * 80)
# print('Used fields:')
# print('-' * 80)
# for field in sorted(used_filed_names):
#   print('-', field)


def collect_all_message_types(message_descriptor, result=None):
  """
    Collects all unique message types within a protobuf message and its nested types.

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


def get_field_type(field):
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


# First, collect all unique message types
all_message_types = set()
for msg_type in [
    sp.capabilities_pb2.Capabilities,
    sp.extended_expression_pb2.ExtendedExpression, sp.plan_pb2.Plan,
    sp.function_pb2.FunctionSignature
]:
  collect_all_message_types(msg_type.DESCRIPTOR, all_message_types)

# # Then, list the fields for each message type
# print('-' * 80)
# print('All fields:')
# print('-' * 80)
# for msg_type in all_message_types:
#   for field in msg_type.fields:
#     print(f"- {full_field_name(field)} ({get_field_type(field)})")

all_fields = (
    field for msg_type in all_message_types for field in msg_type.fields)

# Load into data frames.
df_all = pd.DataFrame(data=all_fields, columns=["field"])
df_used = pd.DataFrame(data=used_fields, columns=["field"])

# print("Number of all fields:", len(df_all.index))
# print("Number of used fields:", len(df_used.index))

df_all["full_name"] = df_all["field"].apply(full_field_name)
df_used["full_name"] = df_used["field"].apply(full_field_name)

df_used["covered"] = 1
df_used = df_used.drop("field", axis="columns")

df = pd.merge(df_all, df_used, on="full_name", how="left")
df.covered = df.covered.fillna(0).astype(int)

# Compute useful columns.
df["file"] = df["field"].apply(lambda x: x.file.name)
df["type"] = df["field"].apply(get_field_type)
df["name"] = df["field"].apply(lambda x: x.name)

df = df[df.file.str.startswith("proto/")]

df["message"] = df["field"].apply(get_field_container)
df["message_name"] = df["message"].apply(full_message_type_name)
message_names = df["message_name"].apply(lambda x: x.split("."))
depth = message_names.apply(len).max()
tuples = message_names.apply(lambda x: x + [""] * (depth - len(x))).apply(tuple)
msg_level_names = list(f"msg_lvl{i}" for i in range(depth))
idx = pd.MultiIndex.from_tuples(tuples, names=msg_level_names)
df = df.set_index(idx)

df = df.drop(columns=["field", "message", "full_name"], axis="columns")


def join_non_empty(l):
  return ".".join((x for x in l if x))


for i in range(len(msg_level_names) + 1):
  groupby = ["file"] + msg_level_names[0:i]
  df_g = df.reset_index().groupby(groupby).aggregate({
      "covered": ["sum", "count"],
  })
  df_g = df_g.reset_index()
  df_g["fraction"] = df_g["covered"]["sum"] / df_g["covered"]["count"]
  df_g = df_g.sort_values(["fraction"] + groupby,
                          ascending=[False] + ([True] * len(groupby)))
  df_g["fraction"] = (df_g["fraction"] * 100).astype(int).astype(str) + " %"
  df_g.columns = groupby + ["covered (number)", "total", "covered (fraction)"]
  print(df_g.to_markdown(tablefmt="github", index=False))
  print()
