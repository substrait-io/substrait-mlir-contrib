// RUN: substrait-opt -verify-diagnostics -split-input-file %s

// This file tests that `NullableType` rejects invalid constructions.

// -----

// Check that double-wrapping NullableType<NullableType<T>> is rejected.
// The verifier fires when the type is constructed during parsing.

substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error @+2 {{NullableType cannot wrap another NullableType}}
    %0 = named_table @t1 as ["a"]
        : rel<!substrait.nullable<!substrait.nullable<si32>>>
    yield %0 : rel<!substrait.nullable<!substrait.nullable<si32>>>
  }
}

// -----

// Check that `si32??` does NOT parse as doubly-wrapped; the parser only
// consumes one '?' and the remaining '?' causes a parse error.

substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error @+1 {{expected '>'}}
    %0 = named_table @t1 as ["a"] : rel<si32??>
    yield %0 : rel<si32??>
  }
}

// -----

// Check that applying '?' to an explicit NullableType canonical form
// (!substrait.nullable<si32>?) is caught at parse time, not by the verifier.

substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error @+1 {{NullableType cannot be made nullable again}}
    %0 = named_table @t1 as ["a"] : rel<!substrait.nullable<si32>?>
    yield %0 : rel<!substrait.nullable<si32>?>
  }
}
