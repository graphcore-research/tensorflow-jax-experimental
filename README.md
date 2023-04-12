# :red_circle: **Non-official experimental** :red_circle: IPU XLA TensorFlow/JAX XLA backend

This repository is **a non-official experimental** fork of [Graphcore IPU TensorFlow repository](https://github.com/graphcore/tensorflow) (the latter being a modified version of TensorFlow supporting Graphcore IPUs).

The goal of this repository is to implement the additional PRJT layer on top of Graphcore Poplar XLA backend, the former being necessary to compile and run JAX on IPUs. This is **NOT** an additional non-official TensorFlow version for IPUs.

[Experimental JAX on IPU](https://github.com/graphcore-research/jax-mk2-experimental) is directly pulling the XLA backend source code from this repository, and compiling the proper `jaxlib` Python binary wheel. Independent compilation of this repository using `bazel` is only supported in order to directly test bug fixes or additional features on the IPU XLA backend or PJRT client.

## Compilation

The stable branch requires the following configuration: Ubuntu 20.04, [Graphcore Poplar SDK 3.1](https://www.graphcore.ai/posts/poplar-sdk-3.1-now-available) and [Bazel 5.1.1](https://docs.bazel.build/versions/5.1.1/install.html).

For the development of `jaxlib` on IPU, the targets of interest are:
* IPU Poplar XLA backend: `//tensorflow/compiler/plugin:plugin`
* XLA Python client: `//tensorflow/compiler/xla/python:xla_client`
* IPU PJRT client: `//tensorflow/compiler/plugin/poplar/xla_client:ipu_xla_client`

These targets can be compiled as following:
```bash
export PATH=$HOME/bin:$PATH  # in case of local Bazel install
export TF_POPLAR_BASE=...    # Poplar install path. e.g. /opt/poplar/ or ${POPLAR_SDK_ENABLED}
python configure.py
bazel build --config=monolithic //tensorflow/compiler/plugin/poplar/xla_client:ipu_xla_client
```
Note that the option `--config=monolithic` is here to reflect the compilation configuration of `jaxlib`, which generates a single monolithic shared library.

Additional useful `bazel` parameters:
* `--output_user_root`: Update bazel directory (e.g. for a faster local disk);

## Running unit tests

For the purpose of supporting JAX, here are the test targets of interest:

* Poplar XLA backend (IPU specific) unit tests: `//tensorflow/compiler/plugin/poplar:all_tests`
* XLA general unit tests, using IPU Poplar backend: `//tensorflow/compiler/tests:poplar_tests`
* XLA client unit tests: `//tensorflow/compiler/xla/client/lib:poplar_tests`
* IPU PJRT client unit tests: `//tensorflow/compiler/plugin/poplar/xla_client/tests:all_tests`

All the previous test targets can be run on the IPU model using the following commands:
```bash
bazel test --config=monolithic --jobs=16 --verbose_failures --cache_test_results=no --test_timeout=240,360,900,3600 --test_size_filters=small,medium,large --flaky_test_attempts=1 --test_output=all --test_env='TF_POPLAR_FLAGS=--use_ipu_model --ipu_model_tiles=8 --max_compilation_threads=1 --max_infeed_threads=2' //tensorflow/compiler/plugin/poplar/xla_client/tests:all_tests
```
Using IPU hardware requires an additional `test_env` mapping: `--test_env='IPUOF_VIPU_API_PARTITION_ID=xxx`.
Additional logs can be outputted using: `--test_env='POPLAR_LOG_LEVEL=DEBUG' --test_env='TF_CPP_MIN_LOG_LEVEL=0'`.

Failing unit tests should be documented as a Github ticket.

## Additional documentation

* [Original TensorFlow readme;](README_ORIGINAL.md)

