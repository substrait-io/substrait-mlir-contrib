// RUN: substrait-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK:      substrait.plan version 0 : 42 : 1
// CHECK-SAME:   git_hash "hash" producer "producer" {
// CHECK-NEXT: }

substrait.plan
  version 0 : 42 : 1
  git_hash "hash"
  producer "producer"
  {}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation {
// CHECK-NEXT:     named_table
// CHECK-NEXT:     yield
// CHECK-NEXT:   }
// CHECK-NEXT: }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @foo::@bar as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation {
// CHECK-NEXT:     named_table
// CHECK-NEXT:     yield
// CHECK-NEXT:   }
// CHECK-NEXT:   relation {
// CHECK-NEXT:     named_table
// CHECK-NEXT:     yield
// CHECK-NEXT:   }
// CHECK-NEXT: }

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @foo::@bar as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
  relation {
    %0 = named_table @foo::@bar as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
}

// -----

// CHECK:      substrait.plan
// CHECK-NEXT:   relation as ["x", "y", "z"] {
// CHECK-NEXT:     named_table
// CHECK-NEXT:     yield
// CHECK-NEXT:   }
// CHECK-NEXT: }

substrait.plan version 0 : 42 : 1 {
  relation as ["x", "y", "z"] {
    %0 = named_table @t as ["a", "b", "c"] : tuple<si32, tuple<si32>>
    yield %0 : tuple<si32, tuple<si32>>
  }
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   extension_uri @extension at "http://some.url/with/extensions.yml"
// CHECK-NEXT:   extension_function @function at @extension["somefunc"]
// CHECK-NEXT:   extension_type @type at @extension["sometype"]
// CHECK-NEXT:   extension_type_variation @type_var at @extension["sometypevar"]
// CHECK-NEXT:   extension_uri @other.extension at "http://other.url/with/more/extensions.yml"
// CHECK-NEXT:   extension_function @other.function at @other.extension["someotherfunc"]
// CHECK-NEXT: }

substrait.plan version 0 : 42 : 1 {
  extension_uri @extension at "http://some.url/with/extensions.yml"
  extension_function @function at @extension["somefunc"]
  extension_type @type at @extension["sometype"]
  extension_type_variation @type_var at @extension["sometypevar"]
  extension_uri @other.extension at "http://other.url/with/more/extensions.yml"
  extension_function @other.function at @other.extension["someotherfunc"]
}

// -----

// CHECK:      substrait.plan
// CHECK-SAME:   advanced_extension
// CHECK-SAME:     optimization = "protobuf message" : !substrait.any<"http://some.url/with/type.proto">
// CHECK-SAME:     enhancement = "other protobuf message" : !substrait.any<"http://other.url/with/type.proto">
// CHECK-NEXT: }

substrait.plan version 0 : 42 : 1
    advanced_extension
      optimization = "protobuf message" : !substrait.any<"http://some.url/with/type.proto">
      enhancement = "other protobuf message" : !substrait.any<"http://other.url/with/type.proto">
{}

// -----

// CHECK:      substrait.plan
// CHECK-SAME:   expected_type_urls
// CHECK-SAME:     ["http://some.url/with/type.proto", "http://other.url/with/type.proto"]
// CHECK-NEXT: }

substrait.plan version 0 : 42 : 1
    expected_type_urls
      ["http://some.url/with/type.proto", "http://other.url/with/type.proto"]
{}
