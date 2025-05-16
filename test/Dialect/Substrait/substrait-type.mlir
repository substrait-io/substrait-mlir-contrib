// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-opt -split-input-file %s | substrait-opt -split-input-file \
// RUN: | FileCheck %s

// This file tests the custom `SubstraitType` printer and parser.

// Check round-tripping of short-hand form.

// CHECK-LABEL: substrait.plan
// CHECK:            = cross %{{.*}} x %{{.*}} : rel<si32> x rel<si1>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : <si32>
    %1 = named_table @t2 as ["b"] : <si1>
    %2 = cross %0 x %1 : rel<si32> x rel<si1>
    yield %2 : !substrait.relation<si32, si1>
  }
}

// -----


// Check parsing of full form.

// CHECK-LABEL: substrait.plan
// CHECK:            = cross %{{.*}} x %{{.*}} : rel<si32> x rel<si1>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : <si32>
    %1 = named_table @t2 as ["b"] : <si1>
    %2 = cross %0 x %1 : !substrait.relation<si32> x !substrait.relation<si1>
    yield %2 : !substrait.relation<si32, si1>
  }
}
