// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : <!substrait.time>
// CHECK-NEXT:    yield %0 : !substrait.relation<!substrait.time>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : <!substrait.time>
    yield %0 : !substrait.relation<!substrait.time>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : <!substrait.date>
// CHECK-NEXT:    yield %0 : !substrait.relation<!substrait.date>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : <!substrait.date>
    yield %0 : !substrait.relation<!substrait.date>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b"] : <!substrait.timestamp, !substrait.timestamp_tz>
// CHECK-NEXT:    yield %0 : !substrait.relation<!substrait.timestamp, !substrait.timestamp_tz>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : <!substrait.timestamp, !substrait.timestamp_tz>
    yield %0 : !substrait.relation<!substrait.timestamp, !substrait.timestamp_tz>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : <!substrait.binary>
// CHECK-NEXT:    yield %0 : !substrait.relation<!substrait.binary>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : <!substrait.binary>
    yield %0 : !substrait.relation<!substrait.binary>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : <!substrait.string>
// CHECK-NEXT:    yield %0 : !substrait.relation<!substrait.string>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : <!substrait.string>
    yield %0 : !substrait.relation<!substrait.string>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b"] : <f32, f64>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : <f32, f64>
    yield %0 : !substrait.relation<f32, f64>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b", "c"] : <f32, tuple<f32>>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : <f32, tuple<f32>>
    yield %0 : !substrait.relation<f32, tuple<f32>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b", "c", "d", "e"] : <si1, si8, si16, si32, si64>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c", "d", "e"] : <si1, si8, si16, si32, si64>
    yield %0 : !substrait.relation<si1, si8, si16, si32, si64>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b", "c"] : <si1, tuple<si1>>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : <si1, tuple<si1>>
    yield %0 : !substrait.relation<si1, tuple<si1>>
  }
}
