// RUN: substrait-opt -verify-diagnostics -split-input-file %s

// expected-error@+1 {{precision must be in a range of [0..38] but got 40}}
%0 = substrait.literal #substrait.decimal<"0.42", P = 40, S = 2>

// -----

// expected-error@+1 {{scale must be in a range of [0..P] (P = 3) but got 4}}
%0 = substrait.literal #substrait.decimal<"0.42", P = 3, S = 4>
