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
from tensorflow.compiler.plugin.poplar.xla_client.python.ipu_xla_client import (
    IpuTargetType, IpuDeviceMesh, IpuDeviceMeshManager
)


def skipIfNoIpuHardware():
  if not IpuDeviceMeshManager.has_local_ipu_hardware():
    raise unittest.SkipTest("No IPU hardware locally availabe. Skipping test.")


class IpuPjrtDeviceTest(parameterized.TestCase):
  """Base class for running an XLA Computation through the local client."""

  def setUp(self):
    super(IpuPjrtDeviceTest, self).setUp()

  def testIpuDeviceMeshManager__IpuModelMeshProperties(self):
    manager = IpuDeviceMeshManager.create_ipu_model_manager()
    self.assertEqual(len(manager), 1)
    mesh = manager[0]
    self.assertIsInstance(mesh, IpuDeviceMesh)
    self.assertEqual(mesh.id, 0)
    # Default: IPU mk2 hardware.
    self.assertEqual(mesh.target.type, IpuTargetType.IPU_MODEL)
    self.assertEqual(mesh.target.num_ipus, 1)
    self.assertEqual(mesh.target.num_tiles_per_ipu, 1472)

  def testIpuDeviceMeshManager__IpuDeviceMesh__ConsistentMeshes(self):
    skipIfNoIpuHardware()

    manager = IpuDeviceMeshManager.create_ipu_manager()
    meshes = manager.meshes

    ipu_ids = sorted([m.id for m in meshes if len(m) == 1])
    # Single IPUs should be the first ids.
    self.assertEqual(ipu_ids[0], 0)
    self.assertEqual(ipu_ids[-1], len(ipu_ids) - 1)
    # All meshes are based on single IPUs.
    for m in meshes:
      self.assertLessEqual(set(m.info.ipu_ids), set(ipu_ids))
      self.assertLessEqual(sorted(m.info.ipu_ids), m.info.ipu_ids)

      self.assertEqual(m.info.target.num_ipus, len(m.info.ipu_ids))
      assert m.info.target.arch_info.startswith("ipu")
      if len(m) == 1:
        self.assertEqual(m.id, m.info.ipu_ids[0])

  def testIpuDeviceMeshManager__IpuDeviceMesh__FindMesh(self):
    skipIfNoIpuHardware()

    manager = IpuDeviceMeshManager.create_ipu_manager()
    # Existing IPU mesh.
    mesh = manager.find([0, 1])
    self.assertEqual(mesh.info.ipu_ids, [0, 1])
    # No mesh with [0, 2]
    with self.assertRaises(IndexError):
      manager.find([0, 2])


if __name__ == "__main__":
  absltest.main()
