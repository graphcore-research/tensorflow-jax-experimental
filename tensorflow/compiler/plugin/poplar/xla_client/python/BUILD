load("//tensorflow:tensorflow.bzl", "pybind_extension")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar/xla_client:__subpackages__",
])

# Manual definition of the IPU XLA client Python bindings.
# Required to be able to export additional symbols.
# TODO: understand why default `pybind_extension` remove them.
# TODO: understand which options to use to reduce binary size.
cc_binary(
    name = "ipu_xla_client_pybind.so",
    copts = [
        "-fvisibility=hidden",
        "-fno-strict-aliasing",
        "-Wl,-Bsymbolic",
        "-fexceptions",
        "-fPIC",
        "-flto"
    ],
    srcs = [
        "ipu_xla_client_pybind.cc",
    ],
    features=[
        "-use_header_modules",
        "-parse_headers",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/pjrt:ipu_pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/python:py_client",
        "//tensorflow/compiler/xla/python:types",
        "@pybind11",
    ],
    linkopts = [
        "-Wl,-Bsymbolic",
    ],
    linkshared = True,
)

# Empty IPU XLA extension... just here to make bazel happy!
pybind_extension(
    name = "ipu_xla_extension_pybind",
    srcs = [
        "ipu_xla_extension_pybind.cc",
    ],
    copts = [
    ],
    module_name = "ipu_xla_extension_pybind",
    deps = [
        "//tensorflow/compiler/xla/python:types",
        ":ipu_xla_client_pybind.so",
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
        ":ipu_xla_extension_pybind",
        "//tensorflow/compiler/xla/python:xla_client",
    ],
)
