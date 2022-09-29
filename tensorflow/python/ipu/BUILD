# Description:
# Python support for IPU.
#

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("@local_config_popdist//:build_defs_popdist.bzl", "poprun_py_test")

py_library(
    name = "config_lib",
    srcs = ["config.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:backend_protos_py",
        "//tensorflow/compiler/plugin/poplar/ops:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:poputil_ops_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "ipu_lib",
    srcs = [
        "__init__.py",
        "config.py",
        # "dataset_benchmark.py",
        # "gradient_accumulation.py",
        "ipu_compiler.py",
        # "ipu_estimator.py",
        # "ipu_infeed_queue.py",
        # "ipu_multi_worker_strategy.py",
        # "ipu_outfeed_queue.py",
        # "ipu_pipeline_estimator.py",
        # "ipu_run_config.py",
        # "ipu_session_run_hooks.py",
        # "ipu_strategy.py",
        "loops.py",
        "scopes.py",
        # "serving.py",
        "sharding.py",
        "utils.py",
        # "vertex_edsl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":config_lib",
        "//tensorflow/python/ipu/test_utils:test_utils",
        "//tensorflow/python/ipu/distributed:distributed_lib",
        "//tensorflow/compiler/plugin/poplar/driver:backend_config_proto_py",
        "//tensorflow/compiler/plugin/poplar/ops:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:pop_datastream_ops_py",
        "//tensorflow/compiler/plugin/poplar/ops:sendrecv_ops_py",
        "//tensorflow/compiler/plugin/poplar/tools:tensorflow_weights_extractor_lib",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/estimator:estimator_py",
    ] 
    # + if_horovod(["//tensorflow/python/ipu/horovod:horovod_lib"]),
)
