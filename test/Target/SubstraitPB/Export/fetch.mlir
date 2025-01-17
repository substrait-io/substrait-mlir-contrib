// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN: | substrait-translate -protobuf-to-substrait \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN: | FileCheck %s

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      fetch {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        input {
// CHECK-NEXT:          read {
// CHECK:             offset: 3
// CHECK-NEXT:        count: 5

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = fetch 5 offset 3 from %0 : tuple<si32>
    yield %1 : tuple<si32>
  }
}
