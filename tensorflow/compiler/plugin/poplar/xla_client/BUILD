load("//tensorflow:tensorflow.bzl", "pybind_extension")
load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar/xla_client:__subpackages__",
])

py_library(
    name = "ipu_xla_client",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/python:ipu_xla_client",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "//tensorflow/compiler/plugin/poplar/xla_client/tests:all_tests",
    ],
)
