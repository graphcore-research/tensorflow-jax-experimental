load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow/compiler/xla:xla.bzl", "xla_py_test_deps")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar/xla_client:__subpackages__",
])

xla_test(
    name = "ipu_backend_test",
    srcs = ["ipu_backend_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/ipu_backend",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "assign_default_layout_test",
    size = "small",
    srcs = ["assign_default_layout_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes:assign_default_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "infeed_outfeed_test",
    size = "large",
    srcs = ["infeed_outfeed_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/ipu_backend:ipu_compiler",
        "//tensorflow/compiler/plugin/poplar/xla_client/ipu_backend:ipu_platform",
        "//tensorflow/compiler/plugin/poplar/xla_client/pjrt:ipu_pjrt_client",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:local_client_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "input_output_aliasing_test",
    size = "small",
    srcs = ["input_output_aliasing_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes:input_output_aliasing",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "feed_token_verifier_test",
    size = "small",
    srcs = ["feed_token_verifier_test.cc"],
    backends = ["poplar"],
    copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes:feed_token_verifier",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

py_test(
    name = "ipu_xla_legacy_client_test",
    testonly = 1,
    srcs = [
        "ipu_xla_legacy_client_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/python:ipu_xla_client",
        "//tensorflow/compiler/xla/python:custom_call_for_test",
        "//tensorflow/compiler/xla/python:xla_client",
        "//tensorflow/compiler/xla/python:xla_extension",
        "//tensorflow/compiler/xla/python:xla_client_test",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

py_test(
    name = "ipu_xla_pjrt_client_test",
    testonly = 1,
    srcs = [
        "ipu_xla_pjrt_client_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/python:ipu_xla_client",
        "//tensorflow/compiler/xla/python:custom_call_for_test",
        "//tensorflow/compiler/xla/python:xla_client",
        "//tensorflow/compiler/xla/python:xla_extension",
        "//tensorflow/compiler/xla/python:xla_client_test",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

py_test(
    name = "ipu_pjrt_client_base_test",
    testonly = 1,
    srcs = [
        "ipu_pjrt_client_base_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/python:ipu_xla_client",
        "//tensorflow/compiler/xla/python:custom_call_for_test",
        "//tensorflow/compiler/xla/python:xla_client",
        "//tensorflow/compiler/xla/python:xla_extension",
        "//tensorflow/compiler/xla/python:xla_client_test",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

py_test(
    name = "ipu_pjrt_device_test",
    testonly = 1,
    srcs = [
        "ipu_pjrt_device_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/plugin/poplar/xla_client/python:ipu_xla_client",
        "//tensorflow/compiler/xla/python:custom_call_for_test",
        "//tensorflow/compiler/xla/python:xla_client",
        "//tensorflow/compiler/xla/python:xla_extension",
        "//tensorflow/compiler/xla/python:xla_client_test",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

test_suite(
    name = "all_tests",
    tests = [
        "assign_default_layout_test",
        "infeed_outfeed_test",
        "input_output_aliasing_test",
        "ipu_backend_test",
        "ipu_xla_pjrt_client_test",
        "ipu_xla_legacy_client_test",
        "ipu_pjrt_client_base_test",
        "ipu_pjrt_device_test",
    ],
)
