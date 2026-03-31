// RUN: substrait-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | substrait-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// Tests exporting `StructType` columns to protobuf.

// Check that struct<si32, string?> is exported with per-field nullability.

// CHECK-LABEL: relations {
// CHECK:         base_schema {
// CHECK-NEXT:      names: "s"
// CHECK-NEXT:      names: "x"
// CHECK-NEXT:      names: "y"
// CHECK-NEXT:      struct {
// CHECK-NEXT:        types {
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              i32 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              string {
// CHECK-NEXT:                nullability: NULLABILITY_NULLABLE
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:      }
// CHECK-NEXT:    }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["s", "x", "y"] : rel<struct<si32, string?>>
    yield %0 : rel<struct<si32, string?>>
  }
}

// -----

// Check that `struct<si32>?` is exported with NULLABILITY_NULLABLE on the
// struct sub-message.

// CHECK-LABEL: relations {
// CHECK:         base_schema {
// CHECK-NEXT:      names: "s"
// CHECK-NEXT:      names: "x"
// CHECK-NEXT:      struct {
// CHECK-NEXT:        types {
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              i32 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_NULLABLE
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:      }
// CHECK-NEXT:    }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["s", "x"] : rel<struct<si32>?>
    yield %0 : rel<struct<si32>?>
  }
}
