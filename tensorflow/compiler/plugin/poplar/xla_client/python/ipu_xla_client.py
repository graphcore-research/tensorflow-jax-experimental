# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Register XLA client for IPU"""
import os
from typing import Any, Dict

# NOTE: changing these imports requires updating `patch_copy_ipu_xla_client_py` function in JAX
from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.plugin.poplar.xla_client.python import ipu_xla_client_pybind as _ipu_xla

# Expose individual pybind11 classes and methods
ipu_xla_client = _ipu_xla
IpuDevice = _ipu_xla.IpuDevice
IpuConfig = _ipu_xla.IpuConfig
IpuPoplarTargetType = _ipu_xla.IpuPoplarTargetType
IpuDeviceMeshManager = _ipu_xla.IpuDeviceMeshManager
IpuDeviceMesh = _ipu_xla.IpuDeviceMesh
IpuDeviceMeshInfo = _ipu_xla.IpuDeviceMeshInfo
create_ipu_device_mesh_manager = _ipu_xla.create_ipu_device_mesh_manager

IpuPjRtDevice = _ipu_xla.IpuPjRtDevice
IpuPjRtOptions = _ipu_xla.IpuPjRtOptions
IpuPjRtClientState = _ipu_xla.IpuPjRtClientState
IpuPjRtMeshState = _ipu_xla.IpuPjRtMeshState
IpuPjRtExecutableRunInfo = _ipu_xla.IpuPjRtExecutableRunInfo
get_ipu_client = _ipu_xla.get_ipu_client

# Backward compatible declarations.
IpuTargetType = _ipu_xla.IpuPoplarTargetType


def parse_bool(v: str) -> bool:
  """Parse bool string value.
  """
  if isinstance(v, bool):
    return v
  true_flags = ("yes", "true", "t", "1")
  false_flags = ("no", "false", "f", "0")

  if v.lower() in true_flags:
    return True
  elif v.lower() in false_flags:
    return False
  else:
    raise ValueError(
        f"Invalid environment value: {v}, "
        f"should be among {true_flags} or {false_flags}."
    )


def parse_ipu_env_flags(environ: Any = None) -> Dict[str, str]:
  """Parse IPU flags from environnment variables.
  """
  # Useful for testing not to use global env. variables!
  if environ is None:
    environ = os.environ
  # Allowed prefix of env. flags to parse.
  prefixes = ["JAX_IPU_", "XLA_IPU_PLATFORM_"]
  all_flags = {}
  for prefix in prefixes:
    flags = {
        k.replace(prefix, "").lower(): v
        for k, v in environ.items()
        if k.startswith(prefix)
    }
    all_flags.update(flags)
  return all_flags


def make_ipu_legacy_config(flags: Dict[str, str]) -> IpuConfig:
  """Create IPU legacy config instance from env. flags.
  """
  ipu_config = _ipu_xla.IpuConfig()
  ipu_config.num_ipus = int(flags.get("device_count", 1))
  ipu_config.always_rearrange_copies_on_the_host = parse_bool(
      flags.get("always_rearrange_copies_on_the_host", False)
  )
  ipu_config.prefetch_data_streams = parse_bool(
      flags.get("prefetch_data_streams", False)
  )
  ipu_config.place_ops_on_io_tiles = parse_bool(
      flags.get("place_ops_on_io_tiles", False)
  )
  ipu_config.num_io_tiles = int(flags.get("num_io_tiles", 0))
  ipu_config.io_tile_available_memory_proportion = float(
      flags.get("io_tile_available_memory_proportion", 0)
  )
  return ipu_config


def make_ipu_pjrt_options(flags: Dict[str, str]) -> IpuPjRtOptions:
  """Create IPU PjRt client options from env. flags.
  """
  opts = IpuPjRtOptions()
  # TODO: support `visible_devices` flag.
  opts.use_ipu_model = parse_bool(flags.get("use_model", opts.use_ipu_model))
  opts.ipu_model_num_tiles = int(flags.get("model_num_tiles", opts.ipu_model_num_tiles))
  opts.execute_on_host_flops_limit = float(
      flags.get("execute_on_host_flops_limit", opts.execute_on_host_flops_limit)
  )
  opts.always_rearrange_copies_on_the_host = parse_bool(
      flags.get("always_rearrange_copies_on_the_host", False)
  )
  opts.prefetch_data_streams = parse_bool(flags.get("prefetch_data_streams", False))
  return opts


def make_ipu_client(environ: Any = None) -> xla_client.Client:
  """Build an IPU PjRt XLA client (based on Poplar execution engine).
  """
  flags = parse_ipu_env_flags(environ)
  asynchronous = True
  # Still use legacy client by default.
  use_legacy_client = parse_bool(flags.get("use_legacy_client", True))
  if use_legacy_client:
    ipu_config = make_ipu_legacy_config(flags)
    return _ipu_xla.get_legacy_ipu_client(asynchronous, ipu_config=ipu_config)

  # IPU PjRt client, with multi IPUs support.
  ipu_options = make_ipu_pjrt_options(flags)
  return get_ipu_client(asynchronous, ipu_options=ipu_options)
