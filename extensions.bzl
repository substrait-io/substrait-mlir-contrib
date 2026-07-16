"""Module extensions for MLIR Substrait dependencies."""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def _substrait_mlir_deps_impl(ctx):
    """Implementation of the `substrait_mlir_deps` module extension."""

    # Use the local repository consumed as git submodule.
    new_local_repository(
        name = "llvm-raw",
        build_file_content = "# empty",
        path = "third_party/llvm-project",
    )

    new_local_repository(
        name = "substrait-cpp",
        build_file = "BUILD.substrait-cpp.bazel",
        path = "third_party/substrait-cpp",
    )

substrait_mlir_deps = module_extension(
    implementation = _substrait_mlir_deps_impl,
)
