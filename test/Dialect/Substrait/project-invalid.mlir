// RUN: substrait-opt -verify-diagnostics -split-input-file %s

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+1 {{'substrait.project' op has output field type whose prefix is different from input field types ('si32' vs 'si1')}}
    %1 = project %0 : rel<si32> -> rel<si1, si32> {
    ^bb0(%arg : !substrait.struct<si32>):
      %42 = literal 42 : si32
      yield %42 : si32
    }
    yield %1 : rel<si1, si32>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<si32, si32>
    // expected-error@+1 {{'substrait.project' op has output field type whose prefix is different from input field types ('si32', 'si32' vs 'si32')}}
    %1 = project %0 : rel<si32, si32> -> rel<si32> {
    ^bb0(%arg : !substrait.struct<si32, si32>):
      %42 = literal 42 : si32
      yield %42 : si32
    }
    yield %1 : rel<si32>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+1 {{'substrait.project' op has output field type whose new fields are different from the yielded operand types ('si1' vs 'si32')}}
    %1 = project %0 : rel<si32> -> rel<si32, si1> {
    ^bb0(%arg : !substrait.struct<si32>):
      %42 = literal 42 : si32
      yield %42 : si32
    }
    yield %1 : rel<si32, si1>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+1 {{'substrait.project' op has 'expressions' region with mismatching argument type (has: '!substrait.struct<si1>', expected: '!substrait.struct<si32>')}}
    %1 = project %0 : rel<si32> -> rel<si32, si1> {
    ^bb0(%arg : !substrait.struct<si1>):
      %3 = field_reference %arg[0] : !substrait.struct<si1>
      yield %3 : si1
    }
    yield %1 : rel<si32, si1>
  }
}
