# ===-- dialect.py - Test for Python bindings of Substrait dialect --------=== #
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #

# RUN: %PYTHON %s | FileCheck %s

from substrait_mlir.dialects import substrait as ss
from substrait_mlir import ir


def run(f):
  print("\nTEST:", f.__name__)
  with ir.Context(), ir.Location.unknown():
    ss.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testSubstraitDialect
@run
def testSubstraitDialect():
  plan = ss.PlanOp(version=(0, 42, 1))
  print(plan)
  # CHECK: substrait.plan


# CHECK-LABEL: TEST: testPlanOp
@run
def testPlanOp():
  plan = ss.PlanOp(0, 42, 1)
  print(plan)
  # CHECK: substrait.plan version 0 : 42 : 1
  plan = ss.PlanOp(version=(0, 42, 1))
  print(plan)
  # CHECK: substrait.plan version 0 : 42 : 1


# CHECK-LABEL: TEST: testNamedTable
@run
def testNamedTable():
  plan = ss.PlanOp(version=(0, 42, 1))

  with ir.InsertionPoint(plan.body):
    plan_rel = ss.PlanRelOp()
    with ir.InsertionPoint(plan_rel.body):
      si32 = ir.IntegerType.get_signed(32)
      result_type = ss.RelationType.get([si32, si32])
      field_names = ir.ArrayAttr.get([ir.StringAttr.get(n) for n in ["a", "b"]])
      named_table = ss.NamedTableOp(result_type, "t", field_names)
      ss.YieldOp([named_table.result])

  print(plan)
  # CHECK: substrait.plan
  # CHECK: relation {
  # CHECK: named_table @t


# CHECK-LABEL: TEST: testAnyType
@run
def testAnyType():
  t = ss.AnyType.get(type_url="type.googleapis.com/MyMessage")
  print(t)
  # CHECK: !substrait.any<"type.googleapis.com/MyMessage">
  assert isinstance(t, ss.AnyType)


# CHECK-LABEL: TEST: testBinaryType
@run
def testBinaryType():
  t = ss.BinaryType.get()
  print(t)
  # CHECK: !substrait.binary
  assert isinstance(t, ss.BinaryType)


# CHECK-LABEL: TEST: testDateType
@run
def testDateType():
  t = ss.DateType.get()
  print(t)
  # CHECK: !substrait.date
  assert isinstance(t, ss.DateType)


# CHECK-LABEL: TEST: testDecimalType
@run
def testDecimalType():
  t = ss.DecimalType.get(precision=38, scale=10)
  print(t)
  # CHECK: !substrait.decimal<38, 10>
  assert isinstance(t, ss.DecimalType)


# CHECK-LABEL: TEST: testFixedBinaryType
@run
def testFixedBinaryType():
  t = ss.FixedBinaryType.get(length=16)
  print(t)
  # CHECK: !substrait.fixed_binary<16>
  assert isinstance(t, ss.FixedBinaryType)


# CHECK-LABEL: TEST: testFixedCharType
@run
def testFixedCharType():
  t = ss.FixedCharType.get(length=32)
  print(t)
  # CHECK: !substrait.fixed_char<32>
  assert isinstance(t, ss.FixedCharType)


# CHECK-LABEL: TEST: testIntervalDaySecondType
@run
def testIntervalDaySecondType():
  t = ss.IntervalDaySecondType.get()
  print(t)
  # CHECK: !substrait.interval_day_second
  assert isinstance(t, ss.IntervalDaySecondType)


# CHECK-LABEL: TEST: testIntervalYearMonthType
@run
def testIntervalYearMonthType():
  t = ss.IntervalYearMonthType.get()
  print(t)
  # CHECK: !substrait.interval_year_month
  assert isinstance(t, ss.IntervalYearMonthType)


# CHECK-LABEL: TEST: testNullableType
@run
def testNullableType():
  si32 = ir.IntegerType.get_signed(32)
  nullable_type = ss.NullableType.get(si32)
  print(nullable_type)
  # CHECK: !substrait.nullable<si32>
  assert isinstance(nullable_type, ss.NullableType)


# CHECK-LABEL: TEST: testRelationType
@run
def testRelationType():
  si32 = ir.IntegerType.get_signed(32)
  result_type = ss.RelationType.get([si32, si32])
  print(result_type)
  # CHECK: !substrait.relation<si32, si32>
  assert isinstance(result_type, ss.RelationType)


# CHECK-LABEL: TEST: testStringType
@run
def testStringType():
  t = ss.StringType.get()
  print(t)
  # CHECK: !substrait.string
  assert isinstance(t, ss.StringType)


# CHECK-LABEL: TEST: testStructType
@run
def testStructType():
  si32 = ir.IntegerType.get_signed(32)
  si64 = ir.IntegerType.get_signed(64)

  # Plain struct with two fields.
  struct_type = ss.StructType.get([si32, si64])
  print(struct_type)
  # CHECK: !substrait.struct<si32, si64>
  assert isinstance(struct_type, ss.StructType)

  # Empty struct.
  empty_struct = ss.StructType.get([])
  print(empty_struct)
  # CHECK: !substrait.struct<>
  assert isinstance(empty_struct, ss.StructType)

  # Nullable field inside a struct.
  nullable_si32 = ss.NullableType.get(si32)
  struct_with_nullable = ss.StructType.get([si32, nullable_si32])
  print(struct_with_nullable)
  # CHECK: !substrait.struct<si32, si32?>
  assert isinstance(struct_with_nullable, ss.StructType)

  # Nested struct.
  inner = ss.StructType.get([si32])
  outer = ss.StructType.get([inner, si64])
  print(outer)
  # CHECK: !substrait.struct<struct<si32>, si64>
  assert isinstance(outer, ss.StructType)


# CHECK-LABEL: TEST: testTimeType
@run
def testTimeType():
  t = ss.TimeType.get()
  print(t)
  # CHECK: !substrait.time
  assert isinstance(t, ss.TimeType)


# CHECK-LABEL: TEST: testTimestampType
@run
def testTimestampType():
  t = ss.TimestampType.get()
  print(t)
  # CHECK: !substrait.timestamp
  assert isinstance(t, ss.TimestampType)


# CHECK-LABEL: TEST: testTimestampTzType
@run
def testTimestampTzType():
  t = ss.TimestampTzType.get()
  print(t)
  # CHECK: !substrait.timestamp_tz
  assert isinstance(t, ss.TimestampTzType)


# CHECK-LABEL: TEST: testUUIDType
@run
def testUUIDType():
  t = ss.UUIDType.get()
  print(t)
  # CHECK: !substrait.uuid
  assert isinstance(t, ss.UUIDType)


# CHECK-LABEL: TEST: testVarCharType
@run
def testVarCharType():
  t = ss.VarCharType.get(length=255)
  print(t)
  # CHECK: !substrait.var_char<255>
  assert isinstance(t, ss.VarCharType)
