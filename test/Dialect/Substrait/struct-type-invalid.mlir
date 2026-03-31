// RUN: substrait-opt -verify-diagnostics -split-input-file %s

// Tests that double-nullable `StructType` is rejected, mirroring
// `nullable-type-invalid.mlir`.

// Check that `struct<si32>??` causes a parse error (only one '?' is consumed).

substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error @+1 {{expected '>'}}
    %0 = named_table @t1 as ["s", "x"] : rel<struct<si32>??>
    yield %0 : rel<struct<si32>??>
  }
}

// -----

// Check that `!substrait.nullable<!substrait.struct<si32>>?` is rejected
// at parse time.

substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error @+2 {{NullableType cannot be made nullable again}}
    %0 = named_table @t1 as ["s", "x"]
        : rel<!substrait.nullable<!substrait.struct<si32>>?>
    yield %0 : rel<!substrait.nullable<!substrait.struct<si32>>?>
  }
}

// -----

// Check that `NullableType<NullableType<StructType>>` is rejected by the
// verifier.

substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error @+2 {{NullableType cannot wrap another NullableType}}
    %0 = named_table @t1 as ["s", "x"]
        : rel<!substrait.nullable<!substrait.nullable<!substrait.struct<si32>>>>
    yield %0
        : rel<!substrait.nullable<!substrait.nullable<!substrait.struct<si32>>>>
  }
}
