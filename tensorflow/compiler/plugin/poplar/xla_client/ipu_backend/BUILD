load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar/xla_client:__subpackages__",
])

poplar_cc_library(
    name = "ipu_platform_id",
    srcs = ["ipu_platform_id.cc"],
    hdrs = ["ipu_platform_id.h"],
    deps = [
        "//tensorflow/stream_executor:platform",
    ],
)

poplar_cc_library(
    name = "ipu_compiler",
    srcs = [
        "ipu_compiler.cc",
    ],
    hdrs = [
        "ipu_compiler.h",
    ],
    deps = [
        ":ipu_platform_id",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_compiler",
        "//tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ipu_executor",
    srcs = [
        "ipu_executor.cc",
    ],
    hdrs = [
        "ipu_executor.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:poplar_executor",
        "//tensorflow/compiler/xla:statusor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ipu_transfer_manager",
    srcs = [
        "ipu_transfer_manager.cc",
    ],
    hdrs = [
        "ipu_transfer_manager.h",
    ],
    deps = [
        ":ipu_platform_id",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ipu_platform",
    srcs = [
        "ipu_platform.cc",
    ],
    hdrs = [
        "ipu_platform.h",
    ],
    deps = [
        ":ipu_executor",
        ":ipu_platform_id",
        "//tensorflow/compiler/plugin/poplar:poplar_lib",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_platform",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ipu_backend",
    deps = [
        ":ipu_compiler",
        ":ipu_executor",
        ":ipu_platform",
        ":ipu_transfer_manager",
    ],
)
