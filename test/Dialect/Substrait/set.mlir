// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK:           %[[V2:.*]] = named_table
// CHECK-NEXT:      %[[V3:.*]] = set unspecified %[[V0]], %[[V1]], %[[V2]]
// CHECK-SAME:        : tuple<si32>
// CHECK-NEXT:      yield %[[V3]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = named_table @t2 as ["c"] : tuple<si32>
    %3 = set unspecified %0, %1, %2: tuple<si32>
    yield %3 : tuple<si32>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = set minus_primary %[[V0]], %[[V1]] 
// CHECK-SAME:        : tuple<si32>
// CHECK-NEXT:      yield %[[V2]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = set minus_primary %0, %1 : tuple<si32>
    yield %2 : tuple<si32>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = set minus_multiset %[[V0]], %[[V1]] 
// CHECK-SAME:        : tuple<si32>
// CHECK-NEXT:      yield %[[V2]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = set minus_multiset %0, %1 : tuple<si32>
    yield %2 : tuple<si32>
  }
}
// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = set intersection_primary %[[V0]], %[[V1]] 
// CHECK-SAME:        : tuple<si32>
// CHECK-NEXT:      yield %[[V2]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = set intersection_primary %0, %1 : tuple<si32>
    yield %2 : tuple<si32>
  }
}
// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = set intersection_multiset %[[V0]], %[[V1]] 
// CHECK-SAME:        : tuple<si32>
// CHECK-NEXT:      yield %[[V2]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = set intersection_multiset %0, %1 : tuple<si32>
    yield %2 : tuple<si32>
  }
}
// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = set union_distinct %[[V0]], %[[V1]] 
// CHECK-SAME:        : tuple<si32>
// CHECK-NEXT:      yield %[[V2]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = set union_distinct %0, %1 : tuple<si32>
    yield %2 : tuple<si32>
  }
}
// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = set union_all %[[V0]], %[[V1]] 
// CHECK-SAME:        : tuple<si32>
// CHECK-NEXT:      yield %[[V2]] : tuple<si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = set union_all %0, %1 : tuple<si32>
    yield %2 : tuple<si32>
  }
}
