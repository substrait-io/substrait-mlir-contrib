// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = fetch %[[V0]], 3, 5
// CHECK-SAME:        : tuple<si32> -> tuple<si32>
// CHECK-NEXT:      yield %[[V1]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = fetch %0, 3, 5 : tuple<si32> -> tuple<si32>
    yield %1 : tuple<si32>
  }
}