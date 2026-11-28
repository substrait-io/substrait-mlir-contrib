// RUN: substrait-translate -substrait-to-protobuf %s --split-input-file --output-split-marker="# ""-----" | FileCheck %s

// CHECK-LABEL: relations {
// CHECK:         rel {
// CHECK:           sort {
// CHECK:             input {
// CHECK:               read {
// CHECK:                 base_schema {
// CHECK:                   names: "a"
// CHECK:                   names: "b"
// CHECK:                   struct {
// CHECK:                     types {
// CHECK:                       i32 {
// CHECK:                         nullability: NULLABILITY_REQUIRED
// CHECK:                       }
// CHECK:                     }
// CHECK:                     types {
// CHECK:                       i32 {
// CHECK:                         nullability: NULLABILITY_REQUIRED
// CHECK:                       }
// CHECK:                     }
// CHECK:                     nullability: NULLABILITY_REQUIRED
// CHECK:                   }
// CHECK:                 }
// CHECK:                 named_table {
// CHECK:                   names: "t1"
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:             sorts {
// CHECK:               expr {
// CHECK:                 selection {
// CHECK:                   direct_reference {
// CHECK:                     struct_field {
// CHECK:                     }
// CHECK:                   }
// CHECK:                   root_reference {
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               direction: SORT_DIRECTION_ASC_NULLS_FIRST
// CHECK:             }
// CHECK:             sorts {
// CHECK:               expr {
// CHECK:                 selection {
// CHECK:                   direct_reference {
// CHECK:                     struct_field {
// CHECK:                       field: 1
// CHECK:                     }
// CHECK:                   }
// CHECK:                   root_reference {
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               direction: SORT_DIRECTION_DESC_NULLS_LAST
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : rel<si32, si32>
    %1 = sort %0 : rel<si32, si32> by {
      ^bb0(%arg0 : tuple<si32, si32>, %arg1 : tuple<si32, si32>):
        %2 = field_reference %arg0[0] : tuple<si32, si32>
        %3 = field_reference %arg1[0] : tuple<si32, si32>
        %4 = sort_field_compare ASC_NULLS_FIRST %2, %3: (si32, si32) -> si8
        yield %4 : si8
      ^bb1(%arg2 : tuple<si32, si32>, %arg3 : tuple<si32, si32>):
        %5 = field_reference %arg2[1] : tuple<si32, si32>
        %6 = field_reference %arg3[1] : tuple<si32, si32>
        %7 = sort_field_compare DESC_NULLS_LAST %5, %6: (si32, si32) -> si8
        yield %7 : si8
    }
    yield %1 : rel<si32, si32>
  }
}

// -----

// CHECK-LABEL: relations {
// CHECK:         rel {
// CHECK:           sort {
// CHECK:             input {
// CHECK:               read {
// CHECK:                 base_schema {
// CHECK:                   names: "a"
// CHECK:                   struct {
// CHECK:                     types {
// CHECK:                       i32 {
// CHECK:                         nullability: NULLABILITY_REQUIRED
// CHECK:                       }
// CHECK:                     }
// CHECK:                     nullability: NULLABILITY_REQUIRED
// CHECK:                   }
// CHECK:                 }
// CHECK:                 named_table {
// CHECK:                   names: "t2"
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:             sorts {
// CHECK:               expr {
// CHECK:                 literal {
// CHECK:                   i32: 100
// CHECK:                 }
// CHECK:               }
// CHECK:               direction: SORT_DIRECTION_ASC_NULLS_FIRST
// CHECK:             }
// CHECK:             sorts {
// CHECK:               expr {
// CHECK:                 selection {
// CHECK:                   direct_reference {
// CHECK:                     struct_field {
// CHECK:                     }
// CHECK:                   }
// CHECK:                   root_reference {
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               direction: SORT_DIRECTION_DESC_NULLS_LAST
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t2 as ["a"] : rel<si32>
    %1 = sort %0 : rel<si32> by {
      ^bb0(%arg0 : tuple<si32>, %arg1 : tuple<si32>):
        %2 = literal 100 : si32
        %3 = sort_field_compare ASC_NULLS_FIRST %2, %2 : (si32, si32) -> si8
        yield %3 : si8
      ^bb1(%arg2 : tuple<si32>, %arg3 : tuple<si32>):
        %4 = field_reference %arg2[0] : tuple<si32>
        %5 = field_reference %arg3[0] : tuple<si32>
        %6 = sort_field_compare DESC_NULLS_LAST %4, %5 : (si32, si32) -> si8
        yield %6 : si8
    }
    yield %1 : rel<si32>
  }
}
