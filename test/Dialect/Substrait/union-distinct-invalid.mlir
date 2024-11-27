// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = union_distinct %[[V0]] u %[[V1]] 
// CHECK-SAME:        : tuple<si32> u tuple<si32> -> tuple<si32>
// CHECK-NEXT:      yield %[[V2]] : tuple<si32>


substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si1>
    // expected-error@+1 {{left and right inputs must have the same field types}}
    %2 = union_distinct %0 u %1 : tuple<si32> u tuple<si32> -> tuple<si32>
    yield %2 : tuple<si32>
  }
}
