// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN: | substrait-translate -protobuf-to-substrait \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN: | FileCheck %s

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      cross {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        left {
// CHECK-NEXT:          read {
// CHECK:             right {
// CHECK-NEXT:          read {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : <si32>
    %1 = named_table @t2 as ["b"] : <si32>
    %2 = cross %0 x %1 : <si32> x <si32>
    yield %2 : !substrait.relation<si32, si32>
  }
}
