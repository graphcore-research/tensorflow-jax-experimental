load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "dataset_benchmark",
    srcs = [
        "dataset_benchmark.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "exporters",
    srcs = [
        "exporters.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "feeds",
    srcs = [
        "feeds.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "host_embedding",
    srcs = [
        "host_embedding.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ops",
    srcs = [],
    deps = [
        ":dataset_benchmark",
        ":exporters",
        ":feeds",
        ":host_embedding",
    ],
)
