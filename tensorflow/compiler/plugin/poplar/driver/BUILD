load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar:__subpackages__",
    "//tensorflow/python/ipu:__subpackages__",
])

tf_proto_library(
    name = "backend_config_proto",
    srcs = [
        "backend_config.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    protodeps = [
        ":option_flag_proto",
        ":threestate_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_proto",
    ],
)

tf_proto_library(
    name = "config_proto",
    srcs = [
        "config.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
    protodeps = [
        ":threestate_proto",
        ":option_flag_proto",
    ],
)

tf_proto_library(
    name = "option_flag_proto",
    srcs = [
        "option_flag.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "pipeline_config_proto",
    srcs = [
        "pipeline_config.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
    protodeps = [
        ":option_flag_proto",
    ],
)

tf_proto_library(
    name = "poplar_executable_proto",
    srcs = [
        "poplar_executable.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
    protodeps = [
        ":poplar_feed_config_proto",
        ":option_flag_proto",
        "//tensorflow/compiler/xla:xla_data_proto",
    ],
)

tf_proto_library(
    name = "poplar_feed_config_proto",
    srcs = [
        "poplar_feed_config.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
)

tf_proto_library(
    name = "threestate_proto",
    srcs = [
        "threestate.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "trace_proto",
    srcs = [
        "trace.proto",
    ],
    cc_api_version = 2,
    j2objc_api_version = 1,
    make_default_target_header_only = True,
)

genrule(
    name = "generate_poplar_version",
    outs = [
        "poplar_version.cc",
    ],
    cmd = "./$(location //tensorflow/compiler/plugin/poplar:poplar_version_generator) > $@",
    tools = ["//tensorflow/compiler/plugin/poplar:poplar_version_generator"],
)

poplar_cc_library(
    name = "compiler_annotations",
    hdrs = [
        "compiler_annotations.h",
    ],
    deps = [
        ":poplar_feed_config_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/passes:allocation_finder",
        "//tensorflow/compiler/plugin/poplar/driver/tools:feed_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:input_output_aliasing_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:host_embedding",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    # JAXlib compilation into a single library.
    alwayslink = True,
)

poplar_cc_library(
    name = "driver_backend",
    srcs = [
        "extended_graph.cc",
        "extended_tensor.cc",
    ],
    hdrs = [
        "driver_types.h",
        "extended_graph.h",
        "extended_program.h",
        "extended_tensor.h",
    ],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

poplar_cc_library(
    name = "compiler_resources",
    srcs = [
    ],
    hdrs = [
        "compiler_resources.h",
        "prng_seed_state.h",
    ],
    deps = [
        ":compiler_annotations",
        ":compiler_information",
        ":driver_backend",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "compiler_information",
    hdrs = [
        "compiler_information.h",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ipu_devices",
    srcs = [
        "ipu_devices.cc",
    ],
    hdrs = [
        "ipu_devices.h",
    ],
    deps = [
        ":backend_config_proto_cc_impl",
        ":config_proto_cc_impl",
        ":option_flag_proto_cc_impl",
        ":threestate_proto_cc_impl",
        ":xla_ipu_common",
        "//tensorflow/compiler/jit:xla_compilation_cache",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/plugin/poplar/driver/tools:flags",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto_cc_impl",
        "//tensorflow/compiler/xla:xla_proto_cc_impl",
        "//tensorflow/compiler/xla/service:hlo_execution_profile_data_cc_impl",
        "//tensorflow/compiler/xla/service:hlo_proto_cc_impl",
        "//tensorflow/stream_executor:dnn_proto_cc_impl",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "poplar_compiler",
    srcs = [
        "poplar_compiler.cc",
    ],
    hdrs = [
        "poplar_compiler.h",
        "poplar_transfer_manager.h",
        "prng_seed_state.h",
    ],
    deps = [
        ":backend_config_proto_cc_impl",
        ":compiler_resources",
        ":config_proto_cc_impl",
        ":poplar_executable",
        ":poplar_feed_config_proto_cc_impl",
        ":poplar_platform_id",
        ":tensor",
        ":trace_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/invariant_passes",
        "//tensorflow/compiler/plugin/poplar/driver/passes",
        "//tensorflow/compiler/plugin/poplar/driver/poplar_passes",
        "//tensorflow/compiler/plugin/poplar/driver/schedulers:clustering_scheduler",
        "//tensorflow/compiler/plugin/poplar/driver/schedulers:liveness_look_ahead_scheduler",
        "//tensorflow/compiler/plugin/poplar/driver/schedulers:post_order_scheduler",
        "//tensorflow/compiler/plugin/poplar/driver/schedulers:shortest_path_scheduler",
        "//tensorflow/compiler/plugin/poplar/driver/schedulers:sync_list_scheduler",
        "//tensorflow/compiler/plugin/poplar/driver/tools:data_initializer",
        "//tensorflow/compiler/plugin/poplar/driver/tools:debug_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:execution_counter_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:feed_info",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_hash",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_matcher",
        "//tensorflow/compiler/plugin/poplar/driver/tools:progress_bar",
        "//tensorflow/compiler/plugin/poplar/driver/tools:tensor_map",
        "//tensorflow/compiler/plugin/poplar/driver/tools:util",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:entry_visitor",
        "//tensorflow/compiler/plugin/poplar/kernels:ops_proto_cc",
        "//tensorflow/compiler/plugin/poplar/vertices:tf_graph_sources",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla/service/spmd:stateful_rng_spmd_partitioner",
        "//tensorflow/compiler/xla/service/spmd:spmd_partitioner",
        "//tensorflow/compiler/xla/service:conditional_canonicalizer",
        "//tensorflow/compiler/xla/service:custom_call_sharding_helper",
        "//tensorflow/compiler/xla/service:sharding_propagation",
        "//tensorflow/compiler/xla/service:sharding_remover",
        "//tensorflow/compiler/xla/service:all_reduce_reassociate",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:dynamic_dimension_inference",
        "//tensorflow/compiler/xla/service:dynamic_index_splitter",
        "//tensorflow/compiler/xla/service:dynamic_padder",
        "//tensorflow/compiler/xla/service:eigh_expander",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:gather_expander",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:map_inliner",
        "//tensorflow/compiler/xla/service:qr_expander",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:sort_simplifier",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_constant_sinking",
        "//tensorflow/compiler/xla/service:while_loop_invariant_code_motion",
        "//tensorflow/compiler/xla/service:while_loop_simplifier",
        "//tensorflow/compiler/xla/service:zero_sized_hlo_elimination",
        "//tensorflow/core/common_runtime:device_mgr",
        "//tensorflow/core/util:version_info",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "poplar_executable",
    srcs = [
        "poplar_executable.cc",
    ],
    hdrs = [
        "poplar_executable.h",
        "poplar_executable_cache.h",
        "poplar_platform.h",
        "xla_ipu_common.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_executable_binary_file.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:tracepoint.h",
    ],
    deps = [
        ":poplar_executable_proto_cc_impl",
        ":tensor",
        ":xla_ipu_common",
        "//third_party/eigen3",
        "//third_party/ipus/tools:poplar_executable_data",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "poplar_executable_cache",
    srcs = [
        "poplar_executable_cache.cc",
    ],
    hdrs = [
        "poplar_executable.h",
        "poplar_executable_cache.h",
    ],
    deps = [
        ":tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_hash",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "poplar_executor",
    srcs = [
        "poplar_executor.cc",
    ],
    hdrs = [
        "poplar_executor.h",
    ],
    deps = [
        ":poplar_executable",
        ":poplar_platform_id",
        "//tensorflow/compiler/plugin/poplar/driver/tools:hlo_hash",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core/util:version_info",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "poplar_platform_id",
    srcs = [
        "poplar_platform_id.cc",
    ],
    hdrs = [
        "poplar_platform_id.h",
    ],
    deps = [
        "//tensorflow/stream_executor:platform",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "poplar_platform",
    srcs = [
        "poplar_platform.cc",
    ],
    hdrs = [
        "poplar_platform.h",
    ],
    deps = [
        ":poplar_executable",
        ":poplar_platform_id",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "poplar_transfer_manager",
    srcs = [
        "poplar_transfer_manager.cc",
    ],
    hdrs = [
        "poplar_transfer_manager.h",
    ],
    deps = [
        ":poplar_executor",
        ":poplar_platform_id",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "poplar_version",
    hdrs = [
        "poplar_version.h",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "prng_seed_state",
    srcs = [
        "prng_seed_state.cc",
    ],
    hdrs = [
        "prng_seed_state.h",
        "//tensorflow/compiler/plugin/poplar/driver/tools:debug_info.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:option_flag_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver:threestate_proto_cc_impl",
        "//tensorflow/compiler/plugin/poplar/driver/tools:alias_info_proto_cc_impl",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc_impl",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc_impl",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "tensor",
    srcs = [
        "tensor.cc",
    ],
    hdrs = [
        "tensor.h",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops:poplar_ops.h",
    ],
    deps = [
        ":driver_backend",
        "//tensorflow/compiler/plugin/poplar/driver/ops:ops_helper",
        "//tensorflow/compiler/plugin/poplar/driver/tools:conv_poplar_util",
        "//tensorflow/compiler/plugin/poplar/driver/tools:conversions",
        "//tensorflow/compiler/plugin/poplar/driver/tools:matmul_util",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:deferred_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:entry_visitor",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:visitor_base",
        "//tensorflow/compiler/plugin/poplar/driver/visitors:visitor_full",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "vertex_templates",
    hdrs = [
        "vertex_templates.h",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "xla_ipu_common",
    srcs = [
        "xla_ipu_common.cc",
    ],
    hdrs = [
        "xla_ipu_common.h",
    ],
    deps = [
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:tf2xla_proto_cc_impl",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/stream_executor/lib",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "xla_ipu_device",
    srcs = [
        "xla_ipu_device.cc",
    ],
    deps = [
        ":ipu_devices",
        ":xla_ipu_common",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_platform",
        "//tensorflow/core/kernels:no_op",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "xla_ipu_backend",
    srcs = [
        "xla_ipu_backend.cc",
    ],
    visibility = ["//tensorflow/compiler/plugin:__subpackages__"],
    deps = [
        ":xla_ipu_common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/stream_executor/lib",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "driver",
    deps = [
        ":backend_config_proto_cc_impl",
        ":compiler_annotations",
        ":compiler_information",
        ":compiler_resources",
        ":config_proto_cc_impl",
        ":ipu_devices",
        ":option_flag_proto_cc_impl",
        ":pipeline_config_proto_cc_impl",
        ":poplar_compiler",
        ":poplar_executable",
        ":poplar_executable_cache",
        ":poplar_executable_proto_cc_impl",
        ":poplar_executor",
        ":poplar_feed_config_proto_cc_impl",
        ":poplar_platform",
        ":poplar_platform_id",
        ":poplar_transfer_manager",
        ":poplar_version",
        ":prng_seed_state",
        ":tensor",
        ":threestate_proto_cc_impl",
        ":trace_proto_cc_impl",
        ":vertex_templates",
        ":xla_ipu_backend",
        ":xla_ipu_common",
        ":xla_ipu_device",
        "//tensorflow/compiler/plugin/poplar/driver/invariant_passes",
        "//tensorflow/compiler/plugin/poplar/driver/ops",
        "//tensorflow/compiler/plugin/poplar/driver/ops:ops_helper",
        "//tensorflow/compiler/plugin/poplar/driver/ops/custom_ops:ops",
        "//tensorflow/compiler/plugin/poplar/driver/passes",
        "//tensorflow/compiler/plugin/poplar/driver/popit_backend",
        "//tensorflow/compiler/plugin/poplar/driver/poplar_passes",
        "//tensorflow/compiler/plugin/poplar/driver/schedulers",
        "//tensorflow/compiler/plugin/poplar/driver/tools",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops",
        "//tensorflow/compiler/plugin/poplar/driver/visitors",
    ],
    alwayslink = True,
)
