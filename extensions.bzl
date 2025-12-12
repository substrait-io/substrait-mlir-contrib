"""Module extensions for MLIR Substrait dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _substrait_mlir_deps_impl(ctx):
    """Implementation of the `substrait_mlir_deps` module extension."""

    # Use the local repository consumed as git submodule.
    new_local_repository(
        name = "llvm-raw",
        build_file_content = "# empty",
        path = "third_party/llvm-project",
    )

    # Optional LLVM dependencies for performance.
    maybe(
        http_archive,
        name = "llvm_zstd",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
        sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
        strip_prefix = "zstd-1.5.2",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "llvm_zlib",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = [
            "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
    )

    # Needed by `nanobind`.
    http_archive(
        name = "robin_map",
        build_file = "@llvm-raw//utils/bazel/third_party_build:robin_map.BUILD",
        sha256 = "a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236",
        strip_prefix = "robin-map-1.3.0",
        url = "https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz",
    )

    # For some reason, using `bazel_dep(name = "nanobind", ...)` lead to linker
    # errors, so we are using the `http_archive` method like MLIR upstream.
    http_archive(
        name = "nanobind",
        build_file = "@llvm-raw//utils/bazel/third_party_build:nanobind.BUILD",
        sha256 = "8ce3667dce3e64fc06bfb9b778b6f48731482362fb89a43da156632266cd5a90",
        strip_prefix = "nanobind-2.9.2",
        url = "https://github.com/wjakob/nanobind/archive/refs/tags/v2.9.2.tar.gz",
    )

    new_local_repository(
        name = "substrait-cpp",
        build_file = "BUILD.substrait-cpp.bazel",
        path = "third_party/substrait-cpp",
    )

substrait_mlir_deps = module_extension(
    implementation = _substrait_mlir_deps_impl,
)
