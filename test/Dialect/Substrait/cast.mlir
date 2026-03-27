// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// Check that cast operations with various failure behaviors round-trip.
// CHECK-LABEL: substrait.plan
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = project %[[V0]] :
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<si32>):
// CHECK-NEXT:      %[[V2:.*]] = literal 42 : si32
// CHECK-NEXT:      %[[V3:.*]] = cast %[[V2]] or return_null : si32 to si32?
// CHECK-NEXT:      %[[V4:.*]] = cast %[[V2]] or throw_exception : si32 to si64
// CHECK-NEXT:      %[[V5:.*]] = cast %[[V2]] or unspecified : si32 to si64
// CHECK-NEXT:      yield %[[V3]], %[[V4]], %[[V5]] : si32?, si64, si64
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = project %0 : rel<si32> -> rel<si32, si32?, si64, si64> {
    ^bb0(%arg : tuple<si32>):
      %2 = literal 42 : si32
      %3 = cast %2 or return_null : si32 to si32?
      %4 = cast %2 or throw_exception : si32 to si64
      %5 = cast %2 or unspecified : si32 to si64
      yield %3, %4, %5 : si32?, si64, si64
    }
    yield %1 : rel<si32, si32?, si64, si64>
  }
}

// -----

// Check that a cast with return_null has a nullable result type, and that
// the round-trip preserves this. This is primarily a CastOp test; the
// nullable result type is required by CastOp::verify().
// CHECK-LABEL: substrait.plan
// CHECK:       cast %{{.*}} or return_null : si32 to si32?

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    %1 = project %0 : rel<si32> -> rel<si32, si32?> {
    ^bb0(%arg : tuple<si32>):
      %2 = literal 42 : si32
      %3 = cast %2 or return_null : si32 to si32?
      yield %3 : si32?
    }
    yield %1 : rel<si32, si32?>
  }
}
