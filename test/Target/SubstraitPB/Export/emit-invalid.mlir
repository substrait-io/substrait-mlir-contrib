// RUN: substrait-translate -verify-diagnostics -split-input-file %s \
// RUN:   -substrait-to-protobuf

// Two subsequent `emit` ops: the export can't deal with that.

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<si1, si32>
    // expected-note@+1 {{op exported to 'input' message}}
    %1 = emit [1, 0] from %0 : rel<si1, si32> -> rel<si32, si1>
    // expected-error@+1 {{'substrait.emit' op has 'input' that already has 'emit' message (try running canonicalization?)}}
    %2 = emit [1, 0] from %1 : rel<si32, si1> -> rel<si1, si32>
    yield %2 : rel<si1, si32>
  }
}
