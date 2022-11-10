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

from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.plugin.poplar.xla_client.python import ipu_xla_client_pybind as _ipu_xla

IpuDevice = _ipu_xla.IpuDevice
IpuTargetType = _ipu_xla.IpuTargetType


def _str2bool(v: str) -> bool:
  true_flags = ("yes", "true", "t", "1")
  false_flags = ("no", "false", "f", "0")

  if v.lower() in true_flags:
    return True
  elif v.lower() in false_flags:
    return False
  else:
    raise ValueError(f"Invalid environment value: {v}, "
                     f"should be among {true_flags} or {false_flags}.")


def make_ipu_client():
  """Build an IPU XLA client (based on Poplar execution engine).
  """
  num_ipus = os.getenv('XLA_IPU_PLATFORM_DEVICE_COUNT')

  ipu_config = _ipu_xla.IpuConfig()
  if num_ipus:
    ipu_config.num_ipus = int(num_ipus)
  else:
    ipu_config.num_ipus = 1
  
  always_rearrange_copies_on_the_host = \
      os.getenv('XLA_IPU_PLATFORM_ALWAYS_RERRANGE_COPIES_ON_THE_HOST')
  if always_rearrange_copies_on_the_host:
    ipu_config.always_rearrange_copies_on_the_host = \
        _str2bool(always_rearrange_copies_on_the_host)

  prefetch_data_streams = \
    os.getenv('XLA_IPU_PLATFORM_PREFETCH_DATA_STREAMS')
  if prefetch_data_streams:
    ipu_config.prefetch_data_streams = \
        _str2bool(prefetch_data_streams)

  # IOTilesConfig
  num_io_tiles = \
      os.getenv('XLA_IPU_PLATFORM_NUM_IO_TILES')
  if num_io_tiles:
    ipu_config.num_io_tiles = int(num_io_tiles)
  else:
    ipu_config.num_io_tiles = 0

  place_ops_on_io_tiles = \
      os.getenv('XLA_IPU_PLATFORM_PLACE_OPS_ON_IO_TILES')
  if place_ops_on_io_tiles:
    ipu_config.place_ops_on_io_tiles = \
        _str2bool(place_ops_on_io_tiles)
  
  if (ipu_config.place_ops_on_io_tiles and ipu_config.num_io_tiles <= 0):
    raise ValueError("Cannot place ops on I/O tiles when "
                     "num_io_tiles <= 0")
  
  io_tile_available_memory_proportion = \
      os.getenv('XLA_IPU_PLATFORM_IO_TILE_AVAILABLE_MEMORY_PROPORTION')
  if io_tile_available_memory_proportion:
    ipu_config.io_tile_available_memory_proportion \
      = float(io_tile_available_memory_proportion)

  return _ipu_xla.get_ipu_client(asynchronous=True, ipu_config=ipu_config)
