// RUN: substrait-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | substrait-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// Check that si32? is exported with nullability: NULLABILITY_NULLABLE.

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              i32 {
// CHECK-NEXT:                nullability: NULLABILITY_NULLABLE
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32?>
    yield %0 : rel<si32?>
  }
}

// -----

// Check that a non-nullable si32 is still exported as NULLABILITY_REQUIRED.

// CHECK-LABEL: relations {
// CHECK:             i32 {
// CHECK-NEXT:          nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:        }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    yield %0 : rel<si32>
  }
}

// -----

// Check that a nullable tuple<si32> is exported with the struct sub-message
// having nullability: NULLABILITY_NULLABLE, covering the exportType path for
// tuple types wrapped in NullableType.

// CHECK-LABEL: relations {
// CHECK:         struct {
// CHECK:           nullability: NULLABILITY_NULLABLE
// CHECK:         }
// CHECK:         nullability: NULLABILITY_REQUIRED

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<tuple<si32>?>
    yield %0 : rel<tuple<si32>?>
  }
}

// -----

// Check that a relation with mixed nullable and non-nullable fields
// (rel<si32?, f64>) exports NULLABILITY_NULLABLE for si32? and
// NULLABILITY_REQUIRED for f64, covering the recursive exportType path
// through a TupleType with heterogeneous nullability.

// CHECK-LABEL: relations {
// CHECK:           types {
// CHECK-NEXT:        i32 {
// CHECK-NEXT:          nullability: NULLABILITY_NULLABLE
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      types {
// CHECK-NEXT:        fp64 {
// CHECK-NEXT:          nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:        }
// CHECK-NEXT:      }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<si32?, f64>
    yield %0 : rel<si32?, f64>
  }
}
