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
    IpuTargetType, IpuDeviceMesh, IpuDeviceMeshManager, create_ipu_device_mesh_manager,
    IpuPjRtOptions
)

# Skipping some tests if no local IPU hardware.
ipu_hw_available = IpuDeviceMeshManager.has_local_ipu_hardware()


class IpuPjrtDeviceTest(parameterized.TestCase):

  def setUp(self):
    super(IpuPjrtDeviceTest, self).setUp()

  def testIpuDeviceMeshManager__IpuModelMeshProperties(self):
    manager = IpuDeviceMeshManager.create_ipu_model_manager(num_tiles=16)
    self.assertEqual(len(manager), 1)
    mesh = manager[0]
    self.assertIsInstance(mesh, IpuDeviceMesh)
    self.assertEqual(mesh.id, 0)
    # Default: IPU mk2 hardware.
    self.assertEqual(mesh.target.type, IpuTargetType.IPU_MODEL)
    self.assertEqual(mesh.target.num_ipus, 1)
    self.assertEqual(mesh.target.num_tiles_per_ipu, 16)

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__IpuDeviceMesh__ConsistentMeshes(self):
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

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__IpuDeviceMesh__FindMesh(self):
    manager = IpuDeviceMeshManager.create_ipu_manager()
    # Existing IPU mesh.
    mesh = manager.find([0, 1])
    self.assertEqual(mesh.info.ipu_ids, [0, 1])
    # No mesh with [0, 2]
    with self.assertRaises(IndexError):
      manager.find([0, 2])

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__create_ipu_device_mesh_manager__ipu_hardware(self):
    ipu_options = IpuPjRtOptions(use_ipu_model=False)
    ipu_manager = create_ipu_device_mesh_manager(ipu_options)

    self.assertIsInstance(ipu_manager, IpuDeviceMeshManager)
    self.assertGreaterEqual(len(ipu_manager), 1)
    self.assertEqual(ipu_manager[0].type, IpuTargetType.IPU)
    self.assertEqual(ipu_manager[0].num_tiles_per_ipu, 1472)
    self.assertEqual(ipu_manager[0].version, "ipu2")

  def testIpuDeviceMeshManager__create_ipu_device_mesh_manager__ipu_model(self):
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True, ipu_model_num_tiles=16, ipu_model_version="ipu21"
    )
    ipu_manager = create_ipu_device_mesh_manager(ipu_options)

    self.assertIsInstance(ipu_manager, IpuDeviceMeshManager)
    self.assertEqual(len(ipu_manager), 1)
    self.assertEqual(ipu_manager[0].type, IpuTargetType.IPU_MODEL)
    self.assertEqual(ipu_manager[0].num_tiles_per_ipu, 16)
    self.assertEqual(ipu_manager[0].version, "ipu21")


if __name__ == "__main__":
  absltest.main()
