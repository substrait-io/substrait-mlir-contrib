// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-opt -split-input-file %s | substrait-opt -split-input-file \
// RUN: | FileCheck %s

// Tests parsing, printing, and round-tripping of `StructType`.

// -----

// CHECK-LABEL: substrait.plan
// CHECK:       named_table @t1 as ["s", "x"] : rel<struct<si32>>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["s", "x"] : rel<struct<si32>>
    yield %0 : rel<struct<si32>>
  }
}

// -----

// Check round-trip of an empty struct (exercises the early-out path in parse).
// CHECK-LABEL: substrait.plan
// CHECK:       named_table @t1 as ["s"] : rel<struct<>>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["s"] : rel<struct<>>
    yield %0 : rel<struct<>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:       named_table @t1 as ["s", "x", "y"] : rel<struct<si32, string?>?>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["s", "x", "y"] : rel<struct<si32, string?>?>
    yield %0 : rel<struct<si32, string?>?>
  }
}

// -----

// Check round-trip of a nested struct (field names are consumed depth-first).
// CHECK-LABEL: substrait.plan
// CHECK:       named_table @t1 as ["outer", "inner", "inner_a", "inner_b", "z"]
// CHECK-SAME:    : rel<struct<struct<si32, si64>, string>>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["outer", "inner", "inner_a", "inner_b", "z"]
        : rel<struct<struct<si32, si64>, string>>
    yield %0 : rel<struct<struct<si32, si64>, string>>
  }
}

// -----

// Check that the canonical `!substrait.struct<...>` form round-trips to
// short form.
// CHECK-LABEL: substrait.plan
// CHECK:       named_table @t1 as ["s", "x"] : rel<struct<si32>>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["s", "x"] : rel<!substrait.struct<si32>>
    yield %0 : rel<!substrait.struct<si32>>
  }
}
