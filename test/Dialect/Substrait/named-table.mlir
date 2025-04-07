// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as [] : tuple<>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as [] : tuple<>
    yield %0 : tuple<>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : tuple<si32>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    yield %0 : tuple<si32>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b"] : tuple<si32, si32>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1
// CHECK-SAME:      as ["outer", "inner"] : tuple<tuple<si32>>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["outer", "inner"] : tuple<tuple<si32>>
    yield %0 : tuple<tuple<si32>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1
// CHECK-SAME:      as ["a", "a"] : tuple<tuple<si32>>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "a"] : tuple<tuple<si32>>
    yield %0 : tuple<tuple<si32>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         named_table @t1 as ["a"]
// CHECK-SAME:        advanced_extension optimization = "\08*"
// CHECK-SAME:          : !substrait.any<"type.googleapis.com/google.protobuf.Int32Value">
// CHECK-SAME:        : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"]
            advanced_extension optimization = "\08*"
              : !substrait.any<"type.googleapis.com/google.protobuf.Int32Value">
            : tuple<si32>
    yield %0 : tuple<si32>
  }
}
