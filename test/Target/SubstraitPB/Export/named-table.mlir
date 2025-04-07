// RUN: substrait-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | substrait-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// CHECK-LABEL: relations  {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        base_schema {
// CHECK-NEXT:          struct {
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {
// CHECK-NEXT:          names: "t1"
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  version

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as [] : tuple<>
    yield %0 : tuple<>
  }
}

// -----

// CHECK-LABEL: relations  {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              i32 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {
// CHECK-NEXT:          names: "t1"
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  version

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    yield %0 : tuple<si32>
  }
}

// -----

// CHECK-LABEL: relations  {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          names: "b"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              i32 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            types {
// CHECK-NEXT:              i32 {
// CHECK-NEXT:                nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {
// CHECK-NEXT:          names: "t1"
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  version

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
}

// -----

// CHECK-LABEL: relations  {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        base_schema {
// CHECK-NEXT:          names: "outer"
// CHECK-NEXT:          names: "inner"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              struct {
// CHECK-NEXT:                types {
// CHECK-NEXT:                  i32 {
// CHECK-NEXT:                    nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {
// CHECK-NEXT:          names: "t1"
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  version

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["outer", "inner"] : tuple<tuple<si32>>
    yield %0 : tuple<tuple<si32>>
  }
}

// -----

// CHECK-LABEL: relations  {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      read {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        base_schema {
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          names: "a"
// CHECK-NEXT:          struct {
// CHECK-NEXT:            types {
// CHECK-NEXT:              struct {
// CHECK-NEXT:                types {
// CHECK-NEXT:                  i32 {
// CHECK-NEXT:                    nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            nullability: NULLABILITY_REQUIRED
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        named_table {
// CHECK-NEXT:          names: "t1"
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  version

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "a"] : tuple<tuple<si32>>
    yield %0 : tuple<tuple<si32>>
  }
}
