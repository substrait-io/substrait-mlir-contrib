// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-opt -split-input-file %s | substrait-opt -split-input-file \
// RUN: | FileCheck %s

// This file tests the `T?` (NullableType) printing and parsing.

// -----

// Check round-trip of si32? in a named_table.
// CHECK-LABEL: substrait.plan
// CHECK:       named_table @t1 as ["a"] : rel<si32?>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32?>
    yield %0 : rel<si32?>
  }
}

// -----

// Check round-trip with multiple nullable and non-nullable fields.
// CHECK-LABEL: substrait.plan
// CHECK:       named_table @t1 as ["a", "b", "c"] : rel<si32?, f64, si64?>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : rel<si32?, f64, si64?>
    yield %0 : rel<si32?, f64, si64?>
  }
}

// -----

// Check that the canonical MLIR form `!substrait.nullable<si32>` is accepted
// as input and printed back as the sugar form `si32?`.
// CHECK-LABEL: substrait.plan
// CHECK:       named_table @t1 as ["a"] : rel<si32?>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.nullable<si32>>
    yield %0 : rel<!substrait.nullable<si32>>
  }
}
