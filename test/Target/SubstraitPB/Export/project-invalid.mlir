// RUN: substrait-translate -verify-diagnostics -split-input-file %s \
// RUN:   -substrait-to-protobuf

// Empty project op: the export can't deal with that.

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : rel<si32>
    // expected-error@+1 {{'substrait.project' op not supported for export: no expressions}}
    %1 = project %0 : rel<si32> -> rel<si32> {
    ^bb0(%arg0: tuple<si32>):
      yield
    }
    yield %1 : rel<si32>
  }
}
