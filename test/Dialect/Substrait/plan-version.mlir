// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK:      substrait.plan_version 0:42:1 git_hash "hash" producer "producer"
substrait.plan_version 0:42:1 git_hash "hash" producer "producer"

// CHECK-NEXT: substrait.plan_version 1:2:3 producer "other producer"{{$}}
substrait.plan_version 1:2:3 producer "other producer"

// CHECK-NEXT: substrait.plan_version 1:33:7 git_hash "other hash"{{$}}
substrait.plan_version 1:33:7 git_hash "other hash"

// CHECK-NEXT: substrait.plan_version 3:2:1{{$}}
substrait.plan_version 3:2:1

// CHECK-NEXT: substrait.plan_version 6:6:6{{$}}
substrait.plan_version 6:6:6 git_hash "" producer ""
