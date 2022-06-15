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


def make_ipu_client():
  """Build an IPU XLA client (based on Poplar execution engine).
  """
  num_ipus = os.getenv('XLA_IPU_PLATFORM_DEVICE_COUNT')

  ipu_config = _ipu_xla.IpuConfig()
  if num_ipus:
    ipu_config.num_ipus = int(num_ipus)
  else:
    ipu_config.num_ipus = 1
  return _ipu_xla.get_ipu_client(asynchronous=True, ipu_config=ipu_config)
