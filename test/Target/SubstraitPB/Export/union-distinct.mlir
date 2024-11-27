// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN: | FileCheck %s

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      set {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        inputs {
// CHECK-NEXT:          read {
// CHECK:             inputs {
// CHECK-NEXT:          read {
// CHECK:             op: SET_OP_UNION_DISTINCT
  
substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = union_distinct %0 u %1 : tuple<si32> u tuple<si32> -> tuple<si32>
    yield %2 : tuple<si32>
  }
}
