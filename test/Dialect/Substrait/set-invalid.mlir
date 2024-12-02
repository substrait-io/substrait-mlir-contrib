// RUN: substrait-opt -verify-diagnostics %s 

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si1>
    // expected-error@+2 {{'substrait.set' op failed to infer returned types}}
    // expected-error@+1 {{all inputs must have the same field types}}
    %2 = set unspecified, %0, %1 : tuple<si32>, tuple<si1> -> tuple<si32>
    yield %2 : tuple<si32>
  }
}
