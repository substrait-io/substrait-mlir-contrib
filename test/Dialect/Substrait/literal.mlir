// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = project %[[V0]] : tuple<si8> -> tuple<si8, si16, si64> {
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<si8>):
// CHECK-NEXT:      %[[V2:.*]] = literal -1 : si16
// CHECK-NEXT:      %[[V3:.*]] = literal 42 : si64
// CHECK-NEXT:      yield %[[V2]], %[[V3]] : si16, si64
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] : tuple<si8, si16, si64>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si8>
    %1 = project %0 : tuple<si8> -> tuple<si8, si16, si64> {
    ^bb0(%arg : tuple<si8>):
      %-1 = literal -1 : si16
      %42 = literal 42 : si64
      yield %-1, %42 : si16, si64
    }
    yield %1 : tuple<si8, si16, si64>
  }
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = project %[[V0]] : tuple<si1> -> tuple<si1, si32, si64> {
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<si1>):
// CHECK-NEXT:      %[[V2:.*]] = literal -1 : si32
// CHECK-NEXT:      %[[V3:.*]] = literal 42 : si64
// CHECK-NEXT:      yield %[[V2]], %[[V3]] : si32, si64
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] : tuple<si1, si32, si64>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si1>
    %1 = project %0 : tuple<si1> -> tuple<si1, si32, si64> {
    ^bb0(%arg : tuple<si1>):
      %-1 = literal -1 : si32
      %42 = literal 42 : si64
      yield %-1, %42 : si32, si64
    }
    yield %1 : tuple<si1, si32, si64>
  }
}
