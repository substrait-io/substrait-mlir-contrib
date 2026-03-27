// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: tuple<si1>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][0] : tuple<si1>
// CHECK-NEXT:        yield %[[V0]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si1>
    %1 = filter %0 : rel<si1> {
    ^bb0(%arg : tuple<si1>):
      %2 = field_reference %arg[0] : tuple<si1>
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
// CHECK-NEXT:      (%[[ARG0:.*]]: tuple<si1, tuple<si1>>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][1, 0] : tuple<si1, tuple<si1>>
// CHECK-NEXT:        yield %[[V0]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : rel<si1, tuple<si1>>
    %1 = filter %0 : rel<si1, tuple<si1>> {
    ^bb0(%arg : tuple<si1, tuple<si1>>):
      %2 = field_reference %arg[1, 0] : tuple<si1, tuple<si1>>
      yield %2 : si1
    }
    yield %1 : rel<si1, tuple<si1>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: tuple<si1, tuple<si1>>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][1] : tuple<si1, tuple<si1>>
// CHECK-NEXT:        %[[V1:.*]] = field_reference %[[V0]][0] : tuple<si1>
// CHECK-NEXT:        yield %[[V1]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : rel<si1, tuple<si1>>
    %1 = filter %0 : rel<si1, tuple<si1>> {
    ^bb0(%arg : tuple<si1, tuple<si1>>):
      %2 = field_reference %arg[1] : tuple<si1, tuple<si1>>
      %3 = field_reference %2[0] : tuple<si1>
      yield %3 : si1
    }
    yield %1 : rel<si1, tuple<si1>>
  }
}

// -----

// Check that field_reference works into a relation with a nullable element
// type (tuple<si32?>). This covers the computeTypeAtPosition path that strips
// the NullableType wrapper before descending into nested types. The block
// argument type uses the canonical form because the MLIR block-arg parser does
// not apply the T? sugar syntax; only the op assembly format does.
// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      project
// CHECK-NEXT:      (%[[ARG0:.*]]: tuple<!substrait.nullable<si32>>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]][0] : tuple<!substrait.nullable<si32>>
// CHECK-NEXT:        yield %[[V0]] : si32?

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32?>
    %1 = project %0 : rel<si32?> -> rel<si32?, si32?> {
    ^bb0(%arg : tuple<!substrait.nullable<si32>>):
      %2 = field_reference %arg[0] : tuple<!substrait.nullable<si32>>
      yield %2 : si32?
    }
    yield %1 : rel<si32?, si32?>
  }
}
