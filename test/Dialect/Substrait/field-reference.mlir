// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: !substrait.struct<si1>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][0] : !substrait.struct<si1>
// CHECK-NEXT:        yield %[[V0]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si1>
    %1 = filter %0 : rel<si1> {
    ^bb0(%arg : !substrait.struct<si1>):
      %2 = field_reference %arg[0] : !substrait.struct<si1>
      yield %2 : si1
    }
    yield %1 : rel<si1>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: !substrait.struct<si1, struct<si1>>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][1, 0] : !substrait.struct<si1, struct<si1>>
// CHECK-NEXT:        yield %[[V0]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : rel<si1, struct<si1>>
    %1 = filter %0 : rel<si1, struct<si1>> {
    ^bb0(%arg : !substrait.struct<si1, struct<si1>>):
      %2 = field_reference %arg[1, 0] : !substrait.struct<si1, struct<si1>>
      yield %2 : si1
    }
    yield %1 : rel<si1, struct<si1>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: !substrait.struct<si1, struct<si1>>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][1] : !substrait.struct<si1, struct<si1>>
// CHECK-NEXT:        %[[V1:.*]] = field_reference %[[V0]][0] : !substrait.struct<si1>
// CHECK-NEXT:        yield %[[V1]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : rel<si1, struct<si1>>
    %1 = filter %0 : rel<si1, struct<si1>> {
    ^bb0(%arg : !substrait.struct<si1, !substrait.struct<si1>>):
      %2 = field_reference %arg[1] : !substrait.struct<si1, struct<si1>>
      %3 = field_reference %2[0] : !substrait.struct<si1>
      yield %3 : si1
    }
    yield %1 : rel<si1, struct<si1>>
  }
}

// -----

// Check that field_reference strips NullableType before descending into a
// nullable element type (struct<si32?>).
// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      project
// CHECK-NEXT:      (%[[ARG0:.*]]: !substrait.struct<si32?>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][0] : !substrait.struct<si32?>
// CHECK-NEXT:        yield %[[V0]] : si32?

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32?>
    %1 = project %0 : rel<si32?> -> rel<si32?, si32?> {
    ^bb0(%arg : !substrait.struct<si32?>):
      %2 = field_reference %arg[0] : !substrait.struct<si32?>
      yield %2 : si32?
    }
    yield %1 : rel<si32?, si32?>
  }
}

// -----

// Check that field_reference into a struct column yields a StructType, and a
// further reference into that StructType works.
// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: !substrait.struct<struct<si32>>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][0]
// CHECK-SAME:            : !substrait.struct<struct<si32>>
// CHECK-NEXT:        %[[V1:.*]] = field_reference %[[V0]][0]
// CHECK-SAME:            : !substrait.struct<si32>
// CHECK-NEXT:        %[[V2:.*]] = literal 0 : si1
// CHECK-NEXT:        yield %[[V2]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["s", "x"] : rel<struct<si32>>
    %1 = filter %0 : rel<struct<si32>> {
    ^bb0(%arg : !substrait.struct<struct<si32>>):
      // Extract the struct column, then descend into it.
      %2 = field_reference %arg[0] : !substrait.struct<struct<si32>>
      %3 = field_reference %2[0] : !substrait.struct<si32>
      %4 = literal 0 : si1
      yield %4 : si1
    }
    yield %1 : rel<struct<si32>>
  }
}

// -----

// Check two-level descent into a nested struct column.
// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: !substrait.struct<struct<si32, si64>>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][0]
// CHECK-SAME:            : !substrait.struct<struct<si32, si64>>
// CHECK-NEXT:        %[[V1:.*]] = field_reference %[[V0]][1]
// CHECK-SAME:            : !substrait.struct<si32, si64>
// CHECK-NEXT:        %[[V2:.*]] = literal 0 : si1
// CHECK-NEXT:        yield %[[V2]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["col", "x", "y"] : rel<struct<si32, si64>>
    %1 = filter %0 : rel<struct<si32, si64>> {
    ^bb0(%arg : !substrait.struct<struct<si32, si64>>):
      %2 = field_reference %arg[0] : !substrait.struct<struct<si32, si64>>
      %3 = field_reference %2[1] : !substrait.struct<si32, si64>
      %4 = literal 0 : si1
      yield %4 : si1
    }
    yield %1 : rel<struct<si32, si64>>
  }
}
