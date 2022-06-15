load("//tensorflow:tensorflow.bzl", "pybind_extension")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar/xla_client:__subpackages__",
])

pybind_extension(
    name = "ipu_xla_client_pybind",
    srcs = [
        "ipu_xla_client_pybind.cc",
    ],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    module_name = "ipu_xla_client_pybind",
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/pjrt:ipu_pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/python:py_client",
        "//tensorflow/compiler/xla/python:types",
        "@pybind11",
    ],
)

py_library(
    name = "ipu_xla_client",
    srcs = [
        "ipu_xla_client.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":ipu_xla_client_pybind",
        "//tensorflow/compiler/xla/python:xla_client",
    ],
)
