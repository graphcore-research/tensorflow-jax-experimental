load("//tensorflow:tensorflow.bzl", "pybind_extension")
load("//tensorflow/compiler/xla:xla.bzl", "xla_py_test_deps")
load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

poplar_cc_library(
    name = "ipu_pjrt_client",
    srcs = [
        "pjrt/ipu_device.cc",
    ],
    hdrs = [
        "pjrt/ipu_device.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:poplar_lib",
        "//tensorflow/compiler/plugin/poplar/xla_client/ipu_backend",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/pjrt:pjrt_stream_executor_client",
        "//tensorflow/compiler/xla/service:platform_util",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)

pybind_extension(
    name = "ipu_xla_client_pybind",
    srcs = [
        "python/ipu_xla_client_pybind.cc",
    ],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    module_name = "ipu_xla_client_pybind",
    deps = [
        ":ipu_pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/python:py_client",
        "//tensorflow/compiler/xla/python:types",
        "@pybind11",
    ],
)

py_library(
    name = "ipu_xla_client",
    srcs = [
        "python/ipu_xla_client.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":ipu_xla_client_pybind",
        "//tensorflow/compiler/xla/python:xla_client",
    ],
)

py_test(
    name = "ipu_xla_client_test",
    testonly = 1,
    srcs = [
        "python/ipu_xla_client_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":ipu_xla_client",
        "//tensorflow/compiler/xla/python:custom_call_for_test",
        "//tensorflow/compiler/xla/python:xla_client",
        "//tensorflow/compiler/xla/python:xla_extension",
        "//tensorflow/compiler/xla/python:xla_client_test",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)
