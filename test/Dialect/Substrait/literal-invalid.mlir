// RUN: substrait-opt -verify-diagnostics -split-input-file %s

// expected-error@+1 {{unsuited attribute for literal value: unit}}
%0 = substrait.literal unit

// -----

// expected-error@+1 {{decimal value has too many digits after the decimal point (3). Expected <=2 as per the type '!substrait.decimal<10, 2>'}}
%d0 = substrait.literal #substrait.decimal<"123.453", P = 10, S = 2>

// -----

// expected-error@+1 {{'12e.45' is not a valid decimal number}}
%d0 = substrait.literal #substrait.decimal<"12e.45", P = 10, S = 2>

// -----

// expected-error@+1 {{decimal value has too many digits (39). Expected <=10 as per the type '!substrait.decimal<10, 1>'}}
%d0 = substrait.literal #substrait.decimal<"11111111111111111111111111111111111111.1", P = 10, S = 1>

// -----

// expected-error@+1 {{decimal value has too many digits (38). Expected <=37 as per the type '!substrait.decimal<37, 1>'}}
%d0 = substrait.literal #substrait.decimal<"1111111111111111111111111111111111113.2", P = 37, S = 1>
