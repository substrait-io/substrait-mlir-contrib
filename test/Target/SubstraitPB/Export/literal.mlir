// RUN: substrait-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | substrait-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      project {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        input {
// CHECK-NEXT:          read {
// CHECK:             expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            time: 200000000
// CHECK-NEXT:          }
// CHECK-NEXT:        }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si1>
    %1 = project %0 : tuple<si1> -> tuple<si1, !substrait.time> {
    ^bb0(%arg : tuple<si1>):
      %time = literal #substrait.time<200000000us> : !substrait.time
      yield %time : !substrait.time
    }
    yield %1 : tuple<si1, !substrait.time> 
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      project {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        input {
// CHECK-NEXT:          read {
// CHECK:             expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            date: 200000000

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

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      project {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        input {
// CHECK-NEXT:          read {
// CHECK:             expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            timestamp: 10000000000
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            timestamp_tz: 10000000000

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

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      project {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        input {
// CHECK-NEXT:          read {
// CHECK:             expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            binary: "4,5,6,7"

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

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      project {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        input {
// CHECK-NEXT:          read {
// CHECK:             expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            string: "hi"

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

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      project {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        input {
// CHECK-NEXT:          read {
// CHECK:             expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            fp32: 35.35
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            fp64: 42.42

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

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      project {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        input {
// CHECK-NEXT:          read {
// CHECK:             expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            boolean: false
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            i8: 2
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:             i16: -1
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            i32: 35
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        expressions {
// CHECK-NEXT:          literal {
// CHECK-NEXT:             i64: 42

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
