#!/usr/bin/env bash

docker build -f utils/bazel/docker/Dockerfile \
             -t substrait-mlir:dev \
             .

docker run -it \
           -v "$(pwd)":"/opt/src/substrait-mlir" \
           -v "${HOME}/.cache/bazel":"/root/.cache/bazel" \
           substrait-mlir:dev
