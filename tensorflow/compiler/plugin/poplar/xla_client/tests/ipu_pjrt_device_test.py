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
    self.assertEqual(manager.type, IpuTargetType.IPU_MODEL)
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

    self.assertEqual(manager.type, IpuTargetType.IPU)
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
    self.assertEqual(manager.type, IpuTargetType.IPU)
    self.assertEqual(mesh.info.ipu_ids, [0, 1])
    self.assertIn(0, mesh)
    self.assertNotIn(2, mesh)
    # No mesh with [0, 2]
    with self.assertRaises(IndexError):
      manager.find([0, 2])
    # Single IPU mesh
    mesh = manager.find(1)
    self.assertEqual(manager.type, IpuTargetType.IPU)
    self.assertEqual(mesh.info.ipu_ids, [1])
    self.assertIn(1, mesh)
    # From XLA device_assignment
    device_assignment = xla_client.DeviceAssignment.create([[1], [0]])
    mesh = manager.find(device_assignment)
    self.assertEqual(manager.type, IpuTargetType.IPU)
    self.assertEqual(mesh.info.ipu_ids, [0, 1])

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__IpuDeviceMesh__Overlaps(self):
    manager = IpuDeviceMeshManager.create_ipu_manager()
    if manager.count(1) < 4:
      raise unittest.SkipTest("Not enough IPUs to run `overlaps` unit test.")

    mesh = manager.find([0, 1])
    self.assertTrue(mesh.overlaps(mesh))
    self.assertTrue(mesh.overlaps(manager.find([0, 1, 2, 3])))
    self.assertFalse(mesh.overlaps(manager.find([2, 3])))
    self.assertTrue(mesh.overlaps(manager.find(0)))

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__IpuDeviceMesh__OverlappingMeshIds(self):
    manager = IpuDeviceMeshManager.create_ipu_manager()
    if manager.count(1) < 4:
      raise unittest.SkipTest("Not enough IPUs to run `overlaps` unit test.")

    meshes = manager.meshes
    # Single IPU overlap?
    self.assertEqual(
        manager.overlapping_mesh_ids(0), [m.id for m in meshes[1:] if 0 in m]
    )
    # Multi IPUs overlap?
    self.assertEqual(
        manager.overlapping_mesh_ids(meshes[-1].id), [m.id for m in meshes[:-1]]
    )
    # Last mesh should overlap with everyone else!
    last_mesh_id = meshes[-1].id
    for m in meshes[:-1]:
      self.assertIn(last_mesh_id, manager.overlapping_mesh_ids(m.id))

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__create_ipu_device_mesh_manager__ipu_hardware(self):
    ipu_options = IpuPjRtOptions(use_ipu_model=False)
    ipu_manager = create_ipu_device_mesh_manager(ipu_options)

    self.assertIsInstance(ipu_manager, IpuDeviceMeshManager)
    self.assertEqual(ipu_manager.type, IpuTargetType.IPU)
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
    self.assertEqual(ipu_manager.type, IpuTargetType.IPU_MODEL)
    self.assertEqual(len(ipu_manager), 1)
    self.assertEqual(ipu_manager[0].type, IpuTargetType.IPU_MODEL)
    self.assertEqual(ipu_manager[0].num_tiles_per_ipu, 16)
    self.assertEqual(ipu_manager[0].version, "ipu21")

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__default_mesh__multi_ipus_hardware(self):
    ipu_manager = create_ipu_device_mesh_manager(IpuPjRtOptions(use_ipu_model=False))
    if len(ipu_manager) <= 3:
      raise unittest.SkipTest("Not enough IPUs to run `default_mesh` unit test.")

    # First IPU mesh of size 1
    m = ipu_manager.default_mesh(1)
    assert m.size == 1
    assert m.id == 0
    # First IPU mesh of size 2
    m = ipu_manager.default_mesh(2)
    assert m.size == 2
    assert m.id == min([m.id for m in ipu_manager.meshes if m.size == 2])

  def testIpuDeviceMeshManager__default_mesh__ipu_model(self):
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True, ipu_model_num_tiles=16, ipu_model_version="ipu21"
    )
    ipu_manager = create_ipu_device_mesh_manager(ipu_options)
    m = ipu_manager.default_mesh(1)
    assert m.id == 0
    # No IPU model mesh of size 2 (i.e. no comms between IPUs supported)
    with self.assertRaises(IndexError):
      ipu_manager.default_mesh(2)

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__from_mesh_id_to_index__proper_result(self):
    ipu_manager = create_ipu_device_mesh_manager(IpuPjRtOptions(use_ipu_model=False))
    # No masking of devices => two list should coincide.
    ids = [m.id for m in ipu_manager]
    indexes = [ipu_manager.from_mesh_id_to_index(v) for v in ids]
    self.assertEqual(indexes, ids)

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__count__ipu_hardware_different_sizes(self):
    ipu_manager = create_ipu_device_mesh_manager(IpuPjRtOptions(use_ipu_model=False))
    # Initial count for single IPU mesh.
    size, count = 1, ipu_manager.count(1)
    while (ipu_manager.count(size * 2)):
      # Number of mesh of size 2x should be half.
      self.assertEqual(ipu_manager.count(size * 2), count // 2)
      size, count = 2 * size, ipu_manager.count(2 * size)

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__attach__no_overlapping_logic(self):
    manager = IpuDeviceMeshManager.create_ipu_manager()
    if manager.count(1) < 4:
      raise unittest.SkipTest("Not enough IPUs to run `attach` unit test.")

    single_ipus = manager.count(1)
    # Should be able to attach all single IPU devices.
    for id in range(single_ipus):
      self.assertTrue(manager.attach(id, force_detach_overlapping=False))
      self.assertTrue(manager.is_attached(id))
    # Fail to attach all multi-ipus meshes.
    for id in range(single_ipus, len(manager)):
      self.assertFalse(manager.attach(id, force_detach_overlapping=False))

    manager.detach_all()
    # Should be able to attach last one (i.e. all devices mesh).
    self.assertTrue(manager.attach(manager.meshes[-1].id))
    self.assertTrue(manager.is_attached(manager.meshes[-1].id))
    # Fail to attach any other IPU mesh!
    for id in range(0, len(manager) - 1):
      self.assertFalse(manager.attach(id, force_detach_overlapping=False))

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuDeviceMeshManager__attach__overlapping_logic(self):
    manager = IpuDeviceMeshManager.create_ipu_manager()
    if manager.count(1) < 4:
      raise unittest.SkipTest("Not enough IPUs to run `attach` unit test.")

    # Should be able to attach all IPU meshes.
    for id in range(len(manager)):
      self.assertTrue(manager.attach(id, force_detach_overlapping=True))
      self.assertTrue(manager.is_attached(id))
    # None attached, except last one.
    for id in range(len(manager) - 1):
      self.assertFalse(manager.is_attached(id))
    manager.detach_all()
    # Non-trivial overlapping config.
    self.assertTrue(manager.attach(0))
    self.assertTrue(manager.attach(1))
    self.assertTrue(manager.attach(2))
    self.assertTrue(manager.attach(3))
    self.assertTrue(
        manager.attach(manager.find([2, 3]).id, force_detach_overlapping=True)
    )
    # Check IPU meshes as expected?
    self.assertTrue(manager.is_attached(0))
    self.assertTrue(manager.is_attached(1))
    self.assertTrue(manager.is_attached(manager.find([2, 3]).id))
    self.assertFalse(manager.is_attached(2))
    self.assertFalse(manager.is_attached(3))


if __name__ == "__main__":
  absltest.main()
