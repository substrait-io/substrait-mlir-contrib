// RUN: substrait-opt -verify-diagnostics -split-input-file %s

// Test that a cast with 'return_null' must have a nullable result type.
substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = project %0 : rel<si32> -> rel<si32, si64> {
    ^bb0(%arg : !substrait.struct<si32>):
      %2 = literal 42 : si32
      // expected-error@+1 {{result type must be nullable (T?) when 'failure_behavior' is 'return_null', but got 'si64'}}
      %3 = cast %2 or return_null : si32 to si64
      yield %3 : si64
    }
    yield %1 : rel<si32, si64>
  }
}

// -----

// Test that a cast with 'throw_exception' may have a non-nullable result type
// (i.e., the constraint only applies to 'return_null').
substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = project %0 : rel<si32> -> rel<si32, si64> {
    ^bb0(%arg : !substrait.struct<si32>):
      %2 = literal 42 : si32
      %3 = cast %2 or throw_exception : si32 to si64
      yield %3 : si64
    }
    yield %1 : rel<si32, si64>
  }
}

// -----

// Test that a cast with 'return_null' and a nullable result type IS accepted.
// This is the positive counterpart of the first test case.
substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = project %0 : rel<si32> -> rel<si32, si64?> {
    ^bb0(%arg : !substrait.struct<si32>):
      %2 = literal 42 : si32
      %3 = cast %2 or return_null : si32 to si64?
      yield %3 : si64?
    }
    yield %1 : rel<si32, si64?>
  }
}
