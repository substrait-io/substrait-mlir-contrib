// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [1, 0] from %[[V0]] :
// CHECK-SAME:          rel<si1, si32> -> rel<si32, si1>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<si1, si32>
    %1 = emit [1, 0] from %0 : rel<si1, si32> -> rel<si32, si1>
    yield %1 : rel<si32, si1>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [0, 0] from %[[V0]] :
// CHECK-SAME:          rel<si32> -> rel<si32, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = emit [0, 0] from %0 : rel<si32> -> rel<si32, si32>
    yield %1 : rel<si32, si32>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [1] from %[[V0]] : rel<si32, si1> -> rel<si1>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<si32, si1>
    %1 = emit [1] from %0 : rel<si32, si1> -> rel<si1>
    yield %1 : rel<si1>
  }
}
