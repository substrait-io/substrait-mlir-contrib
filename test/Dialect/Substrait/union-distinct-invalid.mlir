// RUN: substrait-opt -verify-diagnostics -split-input-file %s 

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si1>
    // expected-error@+2 {{'substrait.union_distinct' op failed to infer returned types}}
    // expected-error@+1 {{left and right inputs must have the same field types}}
    %2 = union_distinct %0 u %1 : tuple<si32> u tuple<si1> -> tuple<si32>
    yield %2 : tuple<si32>
  }
}
