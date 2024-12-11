// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = join unspecified %[[V0]] j %[[V1]] 
// CHECK-SAME:        : tuple<si32> j tuple<si32> 
// CHECK-NEXT:      yield %[[V2]] : tuple<si32, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = join unspecified %0 j %1 : tuple<si32> j tuple<si32> 
    yield %2 : tuple<si32, si32>
  }
}

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = join inner %[[V0]] j %[[V1]] 
// CHECK-SAME:        : tuple<si32> j tuple<si1> 
// CHECK-NEXT:      yield %[[V2]] : tuple<si32, si1>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si1>
    %2 = join inner %0 j %1 : tuple<si32> j tuple<si1> 
    yield %2 : tuple<si32, si1>
  }
}

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = join outer %[[V0]] j %[[V1]] 
// CHECK-SAME:        : tuple<si32> j tuple<si32> 
// CHECK-NEXT:      yield %[[V2]] : tuple<si32, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = join outer %0 j %1 : tuple<si32> j tuple<si32> 
    yield %2 : tuple<si32, si32>
  }
}

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = join left %[[V0]] j %[[V1]] 
// CHECK-SAME:        : tuple<si32> j tuple<si32> 
// CHECK-NEXT:      yield %[[V2]] : tuple<si32, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = join left %0 j %1 : tuple<si32> j tuple<si32> 
    yield %2 : tuple<si32, si32>
  }
}

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = join right %[[V0]] j %[[V1]] 
// CHECK-SAME:        : tuple<si32> j tuple<si32> 
// CHECK-NEXT:      yield %[[V2]] : tuple<si32, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = join right %0 j %1 : tuple<si32> j tuple<si32> 
    yield %2 : tuple<si32, si32>
  }
}

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = join semi %[[V0]] j %[[V1]] 
// CHECK-SAME:        : tuple<si32> j tuple<si32> 
// CHECK-NEXT:      yield %[[V2]] : tuple<si32, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = join semi %0 j %1 : tuple<si32> j tuple<si32> 
    yield %2 : tuple<si32, si32>
  }
}

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = join anti %[[V0]] j %[[V1]] 
// CHECK-SAME:        : tuple<si32> j tuple<si32> 
// CHECK-NEXT:      yield %[[V2]] : tuple<si32, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = join anti %0 j %1 : tuple<si32> j tuple<si32> 
    yield %2 : tuple<si32, si32>
  }
}

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = join single %[[V0]] j %[[V1]] 
// CHECK-SAME:        : tuple<si32> j tuple<si32> 
// CHECK-NEXT:      yield %[[V2]] : tuple<si32, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = join single %0 j %1 : tuple<si32> j tuple<si32> 
    yield %2 : tuple<si32, si32>
  }
}
