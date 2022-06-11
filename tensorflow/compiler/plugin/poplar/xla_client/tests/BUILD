load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")

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
