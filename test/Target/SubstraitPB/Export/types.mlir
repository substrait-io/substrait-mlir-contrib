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
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              varchar {
// CHECK-NEXT:                length: 6
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<!substrait.var_char<6>>
    yield %0 : tuple<!substrait.var_char<6>>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              uuid {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<!substrait.uuid>
    yield %0 : tuple<!substrait.uuid>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          names: "b"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              interval_year {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              interval_day {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<!substrait.interval_year_month, !substrait.interval_day_second>
    yield %0 : tuple<!substrait.interval_year_month, !substrait.interval_day_second>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              fixed_char {
// CHECK-NEXT:              length: 5
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<!substrait.fixed_char<5>>
    yield %0 : tuple<!substrait.fixed_char<5>>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              time {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<!substrait.time>
    yield %0 : tuple<!substrait.time>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              date {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<!substrait.date>
    yield %0 : tuple<!substrait.date>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          names: "b"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              timestamp {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              timestamp_tz {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<!substrait.timestamp, !substrait.timestamp_tz>
    yield %0 : tuple<!substrait.timestamp, !substrait.timestamp_tz>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              binary {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<!substrait.binary>
    yield %0 : tuple<!substrait.binary>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              string {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<!substrait.string>
    yield %0 : tuple<!substrait.string>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          names: "b"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              fp32 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              fp64 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<f32, f64>
    yield %0 : tuple<f32, f64>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          names: "b"
// CHECK-NEXT:          names: "c"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              fp32 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              struct {
// CHECK-NEXT:                types {
// CHECK-NEXT:                  fp32 {
// CHECK-NEXT:                    nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : tuple<f32, tuple<f32>>
    yield %0 : tuple<f32, tuple<f32>>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          names: "b"
// CHECK-NEXT:          names: "c"
// CHECK-NEXT:          names: "d"
// CHECK-NEXT:          names: "e"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              bool {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              i8 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              i16 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              i32 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              i64 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c", "d", "e"] : tuple<si1, si8, si16, si32, si64>
    yield %0 : tuple<si1, si8, si16, si32, si64>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK:             base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          names: "b"
// CHECK-NEXT:          names: "c"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              bool {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              struct {
// CHECK-NEXT:                types {
// CHECK-NEXT:                  bool {
// CHECK-NEXT:                    nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : tuple<si1, tuple<si1>>
    yield %0 : tuple<si1, tuple<si1>>
  }
}
