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
        "ipu_pjrt_buffer.cc",
        "ipu_pjrt_client.cc",
        "ipu_pjrt_device.cc",
        "ipu_pjrt_executable.cc",
    ],
    hdrs = [
        "ipu_device_mesh.h",
        "ipu_device.h",
        "ipu_pjrt_buffer.h",
        "ipu_pjrt_client.h",
        "ipu_pjrt_device.h",
        "ipu_pjrt_executable.h",
    ],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/util:version_info",
        "//tensorflow/compiler/plugin/poplar:poplar_lib",
        "//tensorflow/compiler/plugin/poplar/xla_client/ipu_backend",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/pjrt:pjrt_stream_executor_client",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/compiler/xla/python:exceptions",
        "//tensorflow/compiler/xla/service:platform_util",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)
