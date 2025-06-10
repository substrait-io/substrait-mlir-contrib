// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = cross %[[V0]] x %[[V1]] : rel<si32> x rel<si1>
// CHECK-NEXT:      yield %[[V2]] : rel<si32, si1>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = named_table @t2 as ["b"] : rel<si1>
    %2 = cross %0 x %1 : rel<si32> x rel<si1>
    yield %2 : rel<si32, si1>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           cross %{{.*}} x %{{[^ ]*}}
// CHECK-SAME:        advanced_extension optimization = "\08*"
// CHECK-SAME:          : !substrait.any<"type.googleapis.com/google.protobuf.Int32Value">
// CHECK-SAME:        : rel<si32> x rel<si1>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = named_table @t2 as ["b"] : rel<si1>
    %2 = cross %0 x %1
            advanced_extension optimization = "\08*"
              : !substrait.any<"type.googleapis.com/google.protobuf.Int32Value">
            : rel<si32> x rel<si1>
    yield %2 : rel<si32, si1>
  }
}
