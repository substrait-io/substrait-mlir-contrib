// RUN: substrait-opt -verify-diagnostics -split-input-file %s

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+1 {{'substrait.filter' op must have 'condition' region yielding one value (yields 2)}}
    %1 = filter %0 : rel<si32> {
    ^bb0(%arg : !substrait.struct<si32>):
      %2 = literal 0 : si1
      yield %2, %2 : si1, si1
    }
    yield %1 : rel<si32>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+1 {{'substrait.filter' op must have 'condition' region yielding 'si1' (yields 'si32')}}
    %1 = filter %0 : rel<si32> {
    ^bb0(%arg : !substrait.struct<si32>):
      %2 = literal 42 : si32
      yield %2 : si32
    }
    yield %1 : rel<si32>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+1 {{'substrait.filter' op must have 'condition' region taking '!substrait.struct<si32>' as argument (takes no arguments)}}
    %1 = filter %0 : rel<si32> {
      %2 = literal 0 : si1
      yield %2 : si1
    }
    yield %1 : rel<si32>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+1 {{'substrait.filter' op must have 'condition' region taking '!substrait.struct<si32>' as argument (takes '!substrait.struct<>')}}
    %1 = filter %0 : rel<si32> {
    ^bb0(%arg : !substrait.struct<>):
      %2 = literal 0 : si1
      yield %2 : si1
    }
    yield %1 : rel<si32>
  }
}
