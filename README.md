# Substrait Dialect for MLIR

[![CI Status of "Build and Test"](https://github.com/substrait-io/substrait-mlir-contrib/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/substrait-io/substrait-mlir-contrib/actions/workflows/build_and_test.yml)
[![CI Status of "clang-format"](https://github.com/substrait-io/substrait-mlir-contrib/actions/workflows/clang_format.yml/badge.svg)](https://github.com/substrait-io/substrait-mlir-contrib/actions/workflows/clang_format.yml)
[![CI Status of "yapf"](https://github.com/substrait-io/substrait-mlir-contrib/actions/workflows/yapf.yml/badge.svg)](https://github.com/substrait-io/substrait-mlir-contrib/actions/workflows/yapf.yml)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Substrait v0.42.1](https://img.shields.io/badge/substrait-v0.42.1-e92063)](https://github.com/substrait-io/substrait/tree/v0.42.1)

This project consist of building an input/output dialect in
[MLIR](https://mlir.llvm.org/) for [Substrait](https://substrait.io/), the
cross-language serialization format of database query plans (akin to an
intermediate representation/IR for database queries). The immediate goal is to
create common infrastructure that can be used to implement consumers, producers,
optimizers, and transpilers of Substrait; the more transcending goal is to study
the viability of using modern, general-purpose compiler infrastructure to
implement database query compilers.

## License

Licensed under the Apache license with LLVM Exceptions. See [LICENSE](LICENSE)
for more information.

## Contributing

Check out the [the dedicated document](CONTRIBUTING.md) for how to contribute.

## Motivation

Substrait defines a serialization format for data-intensive compute operations
similar to relational algebra as they typically occur in database query plans
and similar systems, i.e., an exchange format for database queries. This allows
to separate the development of user frontends such as dataframe libraries or SQL
dialects (aka "Substrait producers") from that of backends such as database
engines (aka "Substrait consumers") and, thus, to interoperate more easily
between different data processing systems.

While Substrait has significant momentum and finds increasing
[adoption](https://substrait.io/community/powered_by/) in mature systems, it is
only concerned with implementing the *serialization format* of query plans, and
leaves the *handling* of that format and, hence, the *in-memory format* or
*intermediate representation* (IR) of plans up to the systems that adopt it.
This will likely lead to repeated implementation effort for everything else
required to deal with that intermediate representation, including
serialization/desiralization to and from text and other formats, a host-language
representation of the IR such as native classes, error and location tracking,
rewrite engines, rewrite rules, and pass management, common optimizations such
as common sub-expression elimination, and similar.

This project aims to create a base for any system dealing with Substrait by
building a "dialect" for Substrait in [MLIR](https://mlir.llvm.org/). In a way,
it aims to build an *in-memory* format for the concepts defined by Substrait,
for which the latter only describe their *serialization format*. MLIR is a
generic compiler framework providing infrastructure for writing compilers from
any domain, is part of the LLVM ecosystem, and has an [active
community](https://discourse.llvm.org/c/mlir/31) with
[adoption](https://mlir.llvm.org/users/) from researchers and industry across
many domains. It makes it easy to add new IR consisting of domain-specific
operations, types, attributes, etc., which are organized in dialects (either
in-tree and out-of-tree), as well as rewrites, passes, conversions,
translations, etc. on those dialects. Creating a Substrait dialect and a number
of common related transformations in such a mature framework has the potential
to eliminate some of the repeated effort described above and, thus, to ease and
eventually increase adoption of Substrait. By extension, building out a dialect
for Substrait can show that MLIR is a viable base for any database-style query
compiler.

## Target Use Cases

The aim of the Substrait dialect is to support all of the following use cases:

* Implement the **translation** of the IR of a particular system to or from
  Substrait by converting it to or from the Substrait dialect (rather than
  Substrait's protobuf messages) and then use the serialization/deserializing
  routines from this project.
* Use the Substrait dialect as the **sole in-memory format** for the IR of a
  particular system, e.g., parsing some frontend format into its own dialect
  and then converting that into the Substrait dialect for export or converting
  from the Substrait dialect for import and then translating that into an
  execution plan.
* Implement **simplifying and "canonicalizing" transformations** of Substrait
  plans such as common sub-expression elimination, dead code elimination,
  sub-query/common table-expression inlining, selection and projection
  push-down, etc., for example, as part of a producer, consumer, or transpiler.
* Implement **"compatibility rewrites"** that transforms plans that using
  features that are unsupported by a particular consumer into equivalent plans
  using features that it does support, for example, as part of a producer,
  consumer, or transpiler.
* [Stretch] Implement a full-blow *query optimizer* using the dialect for both
  logical and physical plans. It is not clear whether this should be done with
  this dialect or rather one or two additional ones that are specifically
  designed with query optimization in mind.

## Design Rationale

The main objective of the Substrait dialect is to allow handling Substrait plans
in MLIR: it replicates the components of Substrait plans as a dialect in order
to be able to tap into MLIR infrastructure. In the [taxonomy of Niu and
Amini](https://www.youtube.com/watch?v=hIt6J1_E21c&t=795s), this means that the
Substrait dialect is both an "input" and an "output" dialect for Substrait. As
such, there is only little freedom in designing the dialect. To guide the design
of the few remaining choices, we shall follow the following rationale (from most
important to least important):

* Every valid Substrait plan MUST be representable in the dialect.
* Every valid Substrait plan MUST round-trip through the dialect to the same
  plan as the input. This includes names and ordering.
* The import routine MUST be able to report all constraint violations of
  Substrait plans (such as type mismatches, dangling references, etc.).
* The dialect MAY be able to represent programs that do not correspond to valid
  Substrait plans. It MAY be impossible to export those to Substrait. For
  example, this allows to represent DAGs of operators rather than just trees.
* Every valid program in the Substrait dialect that can be exported to Substrait
  MUST round-trip through Substrait to a *semantically* equivalent program but
  MAY be different in terms of names, ordering, used operations, attributes,
  etc.
* The dialect SHOULD be understood easily by anyone familiar with Substrait. In
  particular, the dialect SHOULD use the same terminilogy as the Substrait
  specification wherever applicable.
* The dialect SHOULD follow MLIR conventions, idioms, and best practices.
* The dialect SHOULD reuse types, attributes, operations, and interfaces of
  upstream dialects wherever applicable.
* The dialect SHOULD allow simple optimizations and rewrites of Substrait
  plans without requiring other dialects.
* The serialization of the dialect (aka its "assembly") MAY change over time.
  (In other words, the dialect is not meant as an exchange format between
  systems -- that's what Substrait is for.)

## Features (Inherited by MLIR)

MLIR provides infrastructure for virtually all aspects of writing a compiler.
The following is a list of features that we inherit by using MLIR:

* Mostly declarative approach to defining relations and expressions (via
  [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)/tablegen).
* Documentation generation from declared relations and expressions (via
  [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-documentation)).
* Declarative serialization/parsing to/from human-readable text representation
  (via [custom
  assembly](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format)).
* Syntax high-lighting, auto-complete, as-you-type diagnostics, code navigation,
  etc. for the MLIR text format (via an [LSP
  server](https://mlir.llvm.org/docs/Tools/MLIRLSP/)).
* (Partially declarative) type deduction framework (via [ODS
  constraints](https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints)
  or C++
  [interface](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/InferTypeOpInterface.td)
  implementations).
* (Partially declarative) verification of arbitrary consistency constraints,
  declarative (via [ODS
  constraints](https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints))
  or imperative (via [C++
  verifiers](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-verifier-code)).
* Mostly declarative pass management (via
  [tablegen](https://mlir.llvm.org/docs/PassManagement/#declarative-pass-specification)).
* Versatile infrastructure for pattern-based rewriting (via
  [DRR](https://mlir.llvm.org/docs/DeclarativeRewrites/) and [C++
  classes](https://mlir.llvm.org/docs/PatternRewriter/)).
* Powerful manipulation of imperative handling, creation, and modification of IR
  using [native
  classes](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations)
  for operations, types, and attributes,
  [walkers](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#walkers),
  [builders](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Builders.h),
  (IR) [interfaces](https://mlir.llvm.org/docs/Interfaces/), etc. (via ODS and
  C++ infrastructure).
* Powerful
  [location](https://mlir.llvm.org/docs/Dialects/Builtin/#location-attributes)
  tracking and location-based error reporting.
* Generated [Python bindings](https://mlir.llvm.org/docs/Bindings/Python/) of IR
  components, passes, and generic infrastructure (via ODS).
* Powerful command line argument handling and customizable implementation of
  typical [tools](https://github.com/llvm/llvm-project/tree/main/mlir/tools)
  (`X-opt`, `X-translate`, `X-lsp-server`, ...).
* [Testing infrastructure](https://mlir.llvm.org/getting_started/TestingGuide/)
  that is optimized for compilers (via `lit` and `FileCheck`).
* A collection of [common types and
  attributes](https://mlir.llvm.org/docs/Dialects/Builtin/) as well as
  [dialects](https://mlir.llvm.org/docs/Dialects/) (i.e., operations) for more
  or less generic purposes that can be used in or combined with custom dialects
  and that come with [transformations](https://mlir.llvm.org/docs/Passes/) on
  and [conversions](https://mlir.llvm.org/docs/DialectConversion/) to/from other
  dialects.
* A collection of
  [interfaces](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Interfaces)
  and transformation passes on those interfaces, which allows to extend existing
  transformations to new dialects easily.
* A support library with efficient data structures, platform-independent file
  system abstraction, string utilities, etc. (via
  [MLIR](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Support)
  and
  [LLVM](https://github.com/llvm/llvm-project/tree/main/llvm/include/llvm/Support)
  support libraries).

## Build Instructions

This project builds as part of the LLVM External Projects facility (see
[documentation](https://llvm.org/docs/CMake.html#llvm-related-variables)
for the `LLVM_EXTERNAL_PROJECTS` config setting).

### Prerequisites

You need to have the following software installed and in your `PATH` or
[discoverable](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html)
by CMake:

* `git`
* [`ninja`](<https://ninja-build.org/>)
* [LLVM prerequisites](https://llvm.org/docs/GettingStarted.html#software) and a
  [C/C++ toolchain](https://llvm.org/docs/GettingStarted.html#host-c-toolchain-both-compiler-and-standard-library)
* Protobuf >= 3.12 (compiler, runtime, and headers)

### Define Paths

Define the following environment variables (adapted to your situation), ideally
making them permanent in your `$HOME/.bashrc` or in the `activate` script of
your Python virtual environment (see below):

```bash
export SUBSTRAIT_MLIR_SOURCE_DIR=$HOME/git/substrait-mlir-contrib
export SUBSTRAIT_MLIR_BUILD_DIR=${SUBSTRAIT_MLIR_SOURCE_DIR}/build
```

### Check out Project

In your `$HOME/src` directory, clone this project recursively:

```bash
git clone --recursive \
    https://github.com/substrait-io/substrait-mlir-contrib \
    ${SUBSTRAIT_MLIR_SOURCE_DIR}
```

If you have cloned non-recursively already and every time a submodule is
updated, run the following command inside the cloned repository instead:

```bash
cd ${SUBSTRAIT_MLIR_SOURCE_DIR}
git submodule update --recursive --init
```

### Python Prerequisites

Create a virtual environment, activate it, and install the dependencies from
[`requirements.txt`](requirements.txt):

```bash
python3 -m venv ~/.venv/substrait-mlir
source ~/.venv/substrait-mlir/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ${SUBSTRAIT_MLIR_SOURCE_DIR}/requirements.txt
```

For details, see the documentation of the
[MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/).

Make some paths available in your Python environment by adding the following
lines to the end of `~/.venv/substrait-mlir/bin/activate` (then `source` that
file again):

```bash
export SUBSTRAIT_MLIR_SOURCE_DIR=$HOME/git/substrait-mlir-contrib
export SUBSTRAIT_MLIR_BUILD_DIR=${SUBSTRAIT_MLIR_SOURCE_DIR}/build
export PATH=${SUBSTRAIT_MLIR_BUILD_DIR}/bin:$PATH
```

### Configure and Build Main Project

Run the command below to set up the build system, possibly adapting it to your
needs. For example, you may choose not to compile `clang`, `clang-tools-extra`,
`lld`, and/or the examples to save compilation time, or use a different variant
than `Debug`. Similarly, you may want to set `DLLVM_ENABLE_LLD=OFF` on some Macs
that don't have `lld`.

```bash
cmake \
  -DPython3_EXECUTABLE=$(which python) \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra" \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_EXTERNAL_PROJECTS=substrait_mlir \
  -DLLVM_EXTERNAL_SUBSTRAIT_MLIR_SOURCE_DIR=${SUBSTRAIT_MLIR_SOURCE_DIR} \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_ENABLE_PYTHON_BENCHMARKS=ON \
  -DSUBSTRAIT_MLIR_COMPILE_WARNING_AS_ERROR=ON \
  -S${SUBSTRAIT_MLIR_SOURCE_DIR}/third_party/llvm-project/llvm \
  -B${SUBSTRAIT_MLIR_BUILD_DIR} \
  -G Ninja
```

To build, run:

```bash
cd ${SUBSTRAIT_MLIR_BUILD_DIR} && ninja
```

## Using `substrait-opt` and `substrait-translate`

```bash
substrait-opt --help
substrait-translate --help
```

## Running tests

You can run all tests with the following command:

```bash
cd ${SUBSTRAIT_MLIR_BUILD_DIR} && ninja check-substrait-mlir
```

You may also use `lit` to run a subset of the tests.

```bash
llvm-lit -v ${SUBSTRAIT_MLIR_SOURCE_DIR}/test
llvm-lit -v ${SUBSTRAIT_MLIR_SOURCE_DIR}/test/Target
llvm-lit -v ${SUBSTRAIT_MLIR_SOURCE_DIR}/test/python/dialects/substrait/dialect.py
```

## Diagnostics via LSP servers

The [MLIR LSP Servers](https://mlir.llvm.org/docs/Tools/MLIRLSP/) allows editors
to display as-you-type diagnostics, code navigation, and similar features. In
order to extend this functionality to the dialects from this repository, use
the following LSP server binaries:

```bash
${SUBSTRAIT_MLIR_BUILD_DIR}/bin/mlir-proto-lsp-server
${SUBSTRAIT_MLIR_BUILD_DIR}/bin/tblgen-lsp-server",
${SUBSTRAIT_MLIR_BUILD_DIR}/bin/mlir-pdll-lsp-server
```

In VS Code, this is done via the `mlir.server_path`, `mlir.pdll_server_path`,
and `mlir.tablegen_server_path` properties in `settings.json`.
