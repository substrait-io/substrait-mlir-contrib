// RUN: substrait-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | substrait-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// CHECK:      relations {
// CHECK-NEXT:   rel {
// CHECK-NEXT:     read {
// CHECK-NEXT:       common {
// CHECK-NEXT:         direct {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       base_schema {
// CHECK-NEXT:         names: "a"
// CHECK-NEXT:         struct {
// CHECK-NEXT:           types {
// CHECK-NEXT:             i32 {
// CHECK-NEXT:               nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       extension_table {
// CHECK-NEXT:         detail {
// CHECK-NEXT:           type_url: "some url"
// CHECK-NEXT:           value: "some detail"
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: version {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = extension_table
           "some detail" : !substrait.any<"some url">
           as ["a"] : tuple<si32>
    yield %0 : tuple<si32>
  }
}
