// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN: | substrait-translate -protobuf-to-substrait \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN: | FileCheck %s

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      project {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        input {
// CHECK-NEXT:          read {
// CHECK:             expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            boolean: true
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:             i32: 42

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = project %0 : tuple<si32> -> tuple<si32, si1, si32> {
    ^bb0(%arg : tuple<si32>):
      %true = literal -1 : si1
      %42 = literal 42 : si32
      yield %true, %42 : si1, si32
    }
    yield %1 : tuple<si32, si1, si32>
  }
}
