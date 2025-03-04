// RUN: substrait-opt -verify-diagnostics -split-input-file %s


// expected-error@+1 {{unsuited attribute for literal value: unit}}
%0 = substrait.literal unit

// -----

// expected-error@+1 {{decimal value has incorrect number of digits after the decimal point (3). Expected <=2 as per the type '!substrait.decimal<10, 2>'}}
%d0 = substrait.literal #substrait.decimal<"123.453" : !substrait.decimal<10, 2>>

// -----

// expected-error@+1 {{invalid decimal value: 12e.45}}
%d0 = substrait.literal #substrait.decimal<"12e.45" : !substrait.decimal<10, 2>>

// -----

// expected-error@+1 {{decimal value has too many digits (39). Expected at most 38 digits}}
%d0 = substrait.literal #substrait.decimal<"11111111111111111111111111111111111111.1" : !substrait.decimal<10, 1>>

// -----

// expected-error@+1 {{value must have at most '37' digits as per the type ''!substrait.decimal<37, 1>'' but got 38}}
%d0 = substrait.literal #substrait.decimal<"1111111111111111111111111111111111113.2" : !substrait.decimal<37, 1>>

