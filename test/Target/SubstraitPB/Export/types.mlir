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
// CHECK-NEXT:              decimal {
// CHECK-NEXT:                scale: 2
// CHECK-NEXT:                precision: 12
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<decimal<12, 2>>
    yield %0 : rel<decimal<12, 2>>
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
// CHECK-NEXT:              fixed_binary {
// CHECK-NEXT:                length: 4
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<fixed_binary<4>>
    yield %0 : rel<fixed_binary<4>>
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
    %0 = named_table @t1 as ["a"] : rel<var_char<6>>
    yield %0 : rel<var_char<6>>
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
    %0 = named_table @t1 as ["a"] : rel<fixed_char<5>>
    yield %0 : rel<fixed_char<5>>
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
    %0 = named_table @t1 as ["a"] : rel<uuid>
    yield %0 : rel<uuid>
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
    %0 = named_table @t1 as ["a", "b"] : rel<interval_ym, interval_ds>
    yield %0 : rel<interval_ym, interval_ds>
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
    %0 = named_table @t1 as ["a"] : rel<time>
    yield %0 : rel<time>
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
    %0 = named_table @t1 as ["a"] : rel<date>
    yield %0 : rel<date>
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
    %0 = named_table @t1 as ["a", "b"] : rel<timestamp, timestamp_tz>
    yield %0 : rel<timestamp, timestamp_tz>
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
    %0 = named_table @t1 as ["a"] : rel<binary>
    yield %0 : rel<binary>
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
    %0 = named_table @t1 as ["a"] : rel<string>
    yield %0 : rel<string>
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
    %0 = named_table @t1 as ["a", "b"] : rel<f32, f64>
    yield %0 : rel<f32, f64>
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
    %0 = named_table @t1 as ["a", "b", "c"] : rel<f32, tuple<f32>>
    yield %0 : rel<f32, tuple<f32>>
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
    %0 = named_table @t1 as ["a", "b", "c", "d", "e"] : rel<si1, si8, si16, si32, si64>
    yield %0 : rel<si1, si8, si16, si32, si64>
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
    %0 = named_table @t1 as ["a", "b", "c"] : rel<si1, tuple<si1>>
    yield %0 : rel<si1, tuple<si1>>
  }
}
