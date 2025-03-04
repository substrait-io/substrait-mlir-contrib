// RUN: substrait-opt -verify-diagnostics -split-input-file %s

// expected-error@+1 {{precision must be in a range of [0..38] but got 40}}
%0 = substrait.literal "0.42" : !substrait.decimal<40, 2>

// -----

// expected-error@+1 {{scale must be in a range of [0..P] (P = 3) but got 4}}
%0 = substrait.literal "0.42" : !substrait.decimal<3, 4>
