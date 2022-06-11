load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")

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

test_suite(
    name = "all_tests",
    tests = [
        "assign_default_layout_test",
        "ipu_backend_test",
    ],
)
