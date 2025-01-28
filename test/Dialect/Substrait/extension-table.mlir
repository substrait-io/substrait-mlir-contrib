// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = extension_table
// CHECK-SAME:        "some detail" : !substrait.any<"some url">
// CHECK-SAME:        as ["a"] : tuple<si32>
// CHECK-NEXT:      yield %[[V0]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = extension_table
           "some detail" : !substrait.any<"some url">
           as ["a"] : tuple<si32>
    yield %0 : tuple<si32>
  }
}
