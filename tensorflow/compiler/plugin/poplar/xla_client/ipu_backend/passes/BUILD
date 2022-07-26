load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar/xla_client:__subpackages__",
])

poplar_cc_library(
    name = "assign_default_layout",
    srcs = ["assign_default_layout.cc"],
    hdrs = ["assign_default_layout.h"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/strings",
    ],
)

poplar_cc_library(
    name = "input_output_aliasing",
    srcs = ["input_output_aliasing.cc"],
    hdrs = ["input_output_aliasing.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/strings",
    ],
)

poplar_cc_library(
    name = "feed_token_verifier",
    srcs = ["feed_token_verifier.cc"],
    hdrs = ["feed_token_verifier.h"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla:statusor",
        "@com_google_absl//absl/strings",
    ],
)

poplar_cc_library(
    name = "passes",
    deps = [
        ":assign_default_layout",
        ":input_output_aliasing",
        ":feed_token_verifier",
    ],
)
