// RUN: substrait-opt -verify-diagnostics -split-input-file %s

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+2 {{'substrait.emit' op failed to infer returned types}}
    // expected-error@+1 {{1 is not a valid index into '!substrait.relation<si32>'}}
    %1 = emit [1] from %0 : rel<si32> -> rel<si32>
    yield %1 : !substrait.relation<si32>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+2 {{'substrait.emit' op failed to infer returned types}}
    // expected-error@+1 {{-1 is not a valid index into '!substrait.relation<si32>'}}
    %1 = emit [-1] from %0 : rel<si32> -> rel<si32>
    yield %1 : !substrait.relation<si32>
  }
}
