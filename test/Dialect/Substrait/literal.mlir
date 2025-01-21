// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = project %[[V0]] : tuple<si1> -> tuple<si1, !substrait.date> {
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<si1>):
// CHECK-NEXT:      %[[V2:.*]] = literal #substrait.date<200000000> : !substrait.date
// CHECK-NEXT:      yield %[[V2]] : !substrait.date
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] : tuple<si1, !substrait.date>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si1>
    %1 = project %0 : tuple<si1> -> tuple<si1, !substrait.date> {
    ^bb0(%arg : tuple<si1>):
      %date = literal #substrait.date<200000000> : !substrait.date
      yield %date : !substrait.date
    }
    yield %1 : tuple<si1, !substrait.date> 
  }
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = project %[[V0]] : tuple<si1> -> tuple<si1, !substrait.timestamp, !substrait.timestamp_tz> {
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<si1>):
// CHECK-NEXT:      %[[V2:.*]] = literal #substrait.timestamp<10000000000us> 
// CHECK-NEXT:      %[[V3:.*]] = literal #substrait.timestamp_tz<10000000000us> 
// CHECK-NEXT:      yield %[[V2]], %[[V3]] : !substrait.timestamp, !substrait.timestamp_tz
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] : tuple<si1, !substrait.timestamp, !substrait.timestamp_tz>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si1>
    %1 = project %0 : tuple<si1> -> tuple<si1, !substrait.timestamp, !substrait.timestamp_tz> {
    ^bb0(%arg : tuple<si1>):
      %timestamp = literal #substrait.timestamp<10000000000us> 
      %timestamp_tz = literal #substrait.timestamp_tz<10000000000us>
      yield %timestamp, %timestamp_tz : !substrait.timestamp, !substrait.timestamp_tz
    }
    yield %1 : tuple<si1, !substrait.timestamp, !substrait.timestamp_tz>
  }
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = project %[[V0]] : tuple<si1> -> tuple<si1, !substrait.binary> {
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<si1>):
// CHECK-NEXT:      %[[V2:.*]] = literal "4,5,6,7" : !substrait.binary
// CHECK-NEXT:      yield %[[V2]] : !substrait.binary
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] : tuple<si1, !substrait.binary>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si1>
    %1 = project %0 : tuple<si1> -> tuple<si1, !substrait.binary> {
    ^bb0(%arg : tuple<si1>):
      %bytes = literal "4,5,6,7" : !substrait.binary
      yield %bytes : !substrait.binary
    }
    yield %1 : tuple<si1, !substrait.binary>
  }
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = project %[[V0]] : tuple<si1> -> tuple<si1, !substrait.string> {
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<si1>):
// CHECK-NEXT:      %[[V2:.*]] = literal "hi" : !substrait.string
// CHECK-NEXT:      yield %[[V2]] : !substrait.string
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] : tuple<si1, !substrait.string>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si1>
    %1 = project %0 : tuple<si1> -> tuple<si1, !substrait.string> {
    ^bb0(%arg : tuple<si1>):
      %hi = literal "hi" : !substrait.string
      yield %hi : !substrait.string
    }
    yield %1 : tuple<si1, !substrait.string>
  }
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = project %[[V0]] : tuple<f32> -> tuple<f32, f32, f64> {
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<f32>):
// CHECK-NEXT:      %[[V2:.*]] = literal 3.535000e+01 : f32
// CHECK-NEXT:      %[[V3:.*]] = literal 4.242000e+01 : f64
// CHECK-NEXT:      yield %[[V2]], %[[V3]] : f32, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] : tuple<f32, f32, f64>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<f32>
    %1 = project %0 : tuple<f32> -> tuple<f32, f32, f64> {
    ^bb0(%arg : tuple<f32>):
      %35 = literal 35.35 : f32
      %42 = literal 42.42 : f64
      yield %35, %42 : f32, f64
    }
    yield %1 : tuple<f32, f32, f64>
  }
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = project %[[V0]] : tuple<si1> -> tuple<si1, si1, si8, si16, si32, si64> {
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<si1>):
// CHECK-NEXT:      %[[V2:.*]] = literal 0 : si1
// CHECK-NEXT:      %[[V3:.*]] = literal 2 : si8
// CHECK-NEXT:      %[[V4:.*]] = literal -1 : si16
// CHECK-NEXT:      %[[V5:.*]] = literal 35 : si32
// CHECK-NEXT:      %[[V6:.*]] = literal 42 : si64
// CHECK-NEXT:      yield %[[V2]], %[[V3]], %[[V4]], %[[V5]], %[[V6]] : si1, si8, si16, si32, si64
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] : tuple<si1, si1, si8, si16, si32, si64>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si1>
    %1 = project %0 : tuple<si1> -> tuple<si1, si1, si8, si16, si32, si64> {
    ^bb0(%arg : tuple<si1>):
      %false = literal 0 : si1
      %2 = literal 2 : si8
      %-1 = literal -1 : si16
      %35 = literal 35 : si32
      %42 = literal 42 : si64
      yield %false, %2, %-1, %35, %42 : si1, si8, si16, si32, si64
    }
    yield %1 : tuple<si1, si1, si8, si16, si32, si64>
  }
}
