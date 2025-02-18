// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = cross %[[V0]] x %[[V1]] : <si32> x <si1>
// CHECK-NEXT:      yield %[[V2]] : !substrait.relation<si32, si1>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : <si32>
    %1 = named_table @t2 as ["b"] : <si1>
    %2 = cross %0 x %1 : <si32> x <si1>
    yield %2 : !substrait.relation<si32, si1>
  }
}
