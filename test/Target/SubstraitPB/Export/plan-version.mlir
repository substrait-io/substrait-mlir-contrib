// RUN: substrait-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: substrait-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | substrait-translate -protobuf-to-substrait-plan-version \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | substrait-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// CHECK-LABEL: version {
// CHECK-DAG:     minor_number: 42
// CHECK-DAG:     patch_number: 1
// CHECK-DAG:     git_hash: "hash"
// CHECK-DAG:     producer: "producer"
// CHECK-NEXT:  }

substrait.plan_version 0:42:1 git_hash "hash" producer "producer"

// -----

// CHECK-LABEL: version {
// CHECK-DAG:     major_number: 1
// CHECK-DAG:     minor_number: 2
// CHECK-DAG:     patch_number: 3
// CHECK-DAG:     producer: "other producer"
// CHECK-NEXT:  }

substrait.plan_version 1:2:3 producer "other producer"

// -----

// CHECK-LABEL: version {
// CHECK-DAG:     major_number: 1
// CHECK-DAG:     minor_number: 33
// CHECK-DAG:     patch_number: 7
// CHECK-DAG:     git_hash: "other hash"
// CHECK-NEXT:  }

substrait.plan_version 1:33:7 git_hash "other hash"

// -----

// CHECK-LABEL: version {
// CHECK-DAG:     major_number: 3
// CHECK-DAG:     minor_number: 2
// CHECK-DAG:     patch_number: 1
// CHECK-NEXT:  }

substrait.plan_version 3:2:1

// -----

// CHECK-LABEL: version {
// CHECK-NOT:     git_hash
// CHECK-NOT:     producer
// CHECK:       }

substrait.plan_version 1:2:3 git_hash "" producer ""
