# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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
"""IPU PRJT device API unit tests."""
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.xla.python import xla_extension
from tensorflow.compiler.plugin.poplar.xla_client.python.ipu_xla_client import (
    IpuDeviceMesh, IpuDeviceMeshManager, IpuPjRtOptions, get_ipu_client,
    make_ipu_client, IpuPjRtDevice
)

# Skipping some tests if no local IPU hardware.
ipu_hw_available = IpuDeviceMeshManager.has_local_ipu_hardware()


class IpuPjrtClientBaseTest(parameterized.TestCase):

  def setUp(self):
    super(IpuPjrtClientBaseTest, self).setUp()

  def testIpuPjRtclient__get_ipu_client__base_properties(self):
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True, ipu_model_num_tiles=16, ipu_model_version="ipu21"
    )
    ipu_client = get_ipu_client(True, ipu_options)
    self.assertIsInstance(ipu_client, xla_extension.Client)
    self.assertEqual(ipu_client.process_index(), 0)
    self.assertEqual(ipu_client.host_id(), 0)
    self.assertEqual(ipu_client.platform, "ipu")
    self.assertEqual(ipu_client.platform_version, "ipu_sdk3.1.0 (e12d5f9f01)")
    self.assertEqual(ipu_client.runtime_type, "stream_executor")

  def testIpuPjRtclient__get_ipu_client__ipu_model_local_devices(self):
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True, ipu_model_num_tiles=16, ipu_model_version="ipu21"
    )
    ipu_client = get_ipu_client(True, ipu_options)
    ipu_devices = ipu_client.local_devices()

    self.assertEqual(ipu_client.local_device_count(), 1)
    self.assertEqual(ipu_client.device_count(), 1)
    self.assertEqual(len(ipu_devices), 1)

    ipu_device = ipu_devices[0]
    self.assertIsInstance(ipu_device, IpuPjRtDevice)
    self.assertEqual(ipu_device.id, 0)
    self.assertEqual(ipu_device.num_tiles, 16)
    self.assertEqual(ipu_device.version, "ipu21")

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuPjRtclient__get_ipu_client__ipu_hardware_local_devices(self):
    ipu_client = get_ipu_client(True, IpuPjRtOptions())
    ipu_devices = ipu_client.local_devices()

    self.assertGreater(len(ipu_devices), 0)
    self.assertEqual([d.id for d in ipu_devices], list(range(len(ipu_devices))))
    self.assertEqual({d.num_tiles for d in ipu_devices}, {1472})
    self.assertEqual({d.version for d in ipu_devices}, {"ipu2"})


if __name__ == "__main__":
  absltest.main()
