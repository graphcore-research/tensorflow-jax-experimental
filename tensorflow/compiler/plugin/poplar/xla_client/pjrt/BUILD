load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar/xla_client:__subpackages__",
])

poplar_cc_library(
    name = "ipu_pjrt_client",
    srcs = [
        "ipu_device_mesh.cc",
        "ipu_device.cc",
    ],
    hdrs = [
        "ipu_device_mesh.h",
        "ipu_device.h",
    ],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/util:version_info",
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
