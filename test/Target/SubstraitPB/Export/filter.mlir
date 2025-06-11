// RUN: substrait-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | substrait-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      filter {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK:             input {
// CHECK:             condition {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            boolean: true

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = filter %0 : rel<si32> {
    ^bb0(%arg : tuple<si32>):
      %2 = literal -1 : si1
      yield %2 : si1
    }
    yield %1 : rel<si32>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      filter {
// CHECK:             advanced_extension {
// CHECK-NEXT:          optimization {
// CHECK-NEXT:            type_url: "type.googleapis.com/google.protobuf.Int32Value"
// CHECK-NEXT:            value: "\010*"
// CHECK-NEXT:          }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = filter %0
            advanced_extension optimization = "\08*"
              : !substrait.any<"type.googleapis.com/google.protobuf.Int32Value">
            : rel<si32> {
    ^bb0(%arg : tuple<si32>):
      %2 = literal -1 : si1
      yield %2 : si1
    }
    yield %1 : rel<si32>
  }
}
