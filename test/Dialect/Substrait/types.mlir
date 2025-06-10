// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.decimal<12, 2>>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.decimal<12, 2>>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.decimal<12, 2>>
    yield %0 : !substrait.relation<!substrait.decimal<12, 2>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.fixed_binary<4>>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.fixed_binary<4>>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.fixed_binary<4>>
    yield %0 : !substrait.relation<!substrait.fixed_binary<4>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.var_char<6>>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.var_char<6>>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.var_char<6>>
    yield %0 : !substrait.relation<!substrait.var_char<6>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.fixed_char<5>>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.fixed_char<5>>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.fixed_char<5>>
    yield %0 : !substrait.relation<!substrait.fixed_char<5>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.uuid>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.uuid>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.uuid>
    yield %0 : !substrait.relation<!substrait.uuid>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b"] : rel<!substrait.interval_year_month, !substrait.interval_day_second>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.interval_year_month, !substrait.interval_day_second>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<!substrait.interval_year_month, !substrait.interval_day_second>
    yield %0 : !substrait.relation<!substrait.interval_year_month, !substrait.interval_day_second>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.time>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.time>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.time>
    yield %0 : !substrait.relation<!substrait.time>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.date>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.date>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.date>
    yield %0 : !substrait.relation<!substrait.date>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b"] : rel<!substrait.timestamp, !substrait.timestamp_tz>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.timestamp, !substrait.timestamp_tz>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<!substrait.timestamp, !substrait.timestamp_tz>
    yield %0 : !substrait.relation<!substrait.timestamp, !substrait.timestamp_tz>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.binary>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.binary>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.binary>
    yield %0 : !substrait.relation<!substrait.binary>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.binary>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.binary>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.binary>
    yield %0 : !substrait.relation<!substrait.binary>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a"] : rel<!substrait.string>
// CHECK-NEXT:    yield %[[V0]] : !substrait.relation<!substrait.string>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<!substrait.string>
    yield %0 : !substrait.relation<!substrait.string>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b"] : rel<f32, f64>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<f32, f64>
    yield %0 : !substrait.relation<f32, f64>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b", "c"] : rel<f32, tuple<f32>>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : rel<f32, tuple<f32>>
    yield %0 : !substrait.relation<f32, tuple<f32>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b", "c", "d", "e"] : rel<si1, si8, si16, si32, si64>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c", "d", "e"] : rel<si1, si8, si16, si32, si64>
    yield %0 : !substrait.relation<si1, si8, si16, si32, si64>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table @t1 as ["a", "b", "c"] : rel<si1, tuple<si1>>
// CHECK-NEXT:    yield %[[V0]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : rel<si1, tuple<si1>>
    yield %0 : !substrait.relation<si1, tuple<si1>>
  }
}
