// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = extension_table
// CHECK-SAME:         "\08*" : !substrait.any<"type.googleapis.com/google.protobuf.Int32Value">
// CHECK-SAME:        as ["a"] : tuple<si32>
// CHECK-NEXT:      yield %[[V0]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = extension_table
           "\08*" : !substrait.any<"type.googleapis.com/google.protobuf.Int32Value">
           as ["a"] : tuple<si32>
    yield %0 : tuple<si32>
  }
}
