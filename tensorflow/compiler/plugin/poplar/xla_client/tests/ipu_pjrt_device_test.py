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
    IpuTargetType, IpuDeviceMesh, IpuDeviceMeshManager, create_ipu_device_mesh_manager,
    IpuPjRtOptions
)

# Skipping some tests if no local IPU hardware.
num_ipu_hw_available = IpuDeviceMeshManager.num_ipu_hardware_available()


class IpuModelPjrtDeviceTest(parameterized.TestCase):

  def setUp(self):
    super(IpuModelPjrtDeviceTest, self).setUp()

  def testIpuDeviceMeshManager__IpuModelMeshProperties(self):
    manager = IpuDeviceMeshManager.create_ipu_model_manager(num_tiles=16)
    self.assertEqual(len(manager), 1)
    mesh = manager[0]
    self.assertEqual(manager.type, IpuTargetType.IPU_MODEL)
    self.assertIsInstance(mesh, IpuDeviceMesh)
    self.assertEqual(mesh.id, 0)
    self.assertEqual(mesh.local_hardware_id, 0)
    self.assertEqual(mesh.local_device_index, 0)
    # Default: IPU mk2 hardware.
    self.assertEqual(mesh.target.type, IpuTargetType.IPU_MODEL)
    self.assertEqual(mesh.target.num_ipus, 1)
    self.assertEqual(mesh.target.num_tiles_per_ipu, 16)

  def testIpuDeviceMeshManager__create_ipu_device_mesh_manager__multi_ipus(self):
    manager = IpuDeviceMeshManager.create_ipu_model_manager(num_devices=2, num_tiles=16)
    self.assertEqual(len(manager), 2)
    # No IPU mesh grouping the 2 IPUs together: not supported on IPU model.
    self.assertEqual([m.id for m in manager.meshes], [0, 1])
    self.assertEqual([m.local_hardware_id for m in manager.meshes], [0, 1])
    self.assertEqual([m.local_device_index for m in manager.meshes], [0, 1])

  def testIpuDeviceMeshManager__create_ipu_device_mesh_manager__too_many_multi_ipus(
      self
  ):
    with self.assertRaises(xla_extension.XlaRuntimeError):
      IpuDeviceMeshManager.create_ipu_model_manager(num_devices=3)

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


class IpuHwPjrtDeviceTest(parameterized.TestCase):

  def setUp(self):
    super(IpuHwPjrtDeviceTest, self).setUp()
    # Assuming at minimum a Pod4 for testing.
    if num_ipu_hw_available < 4:
      self.skipTest("Not enough IPU hardware available, skipping test.")
    self.manager = None

  def tearDown(self):
    # Force detaching devices (GC may not have cleaned by the time new test is run).
    if isinstance(self.manager, IpuDeviceMeshManager):
      self.manager.detach_all()

  def testIpuDeviceMeshManager__IpuDeviceMesh__ConsistentMeshes(self):
    self.manager = IpuDeviceMeshManager.create_ipu_manager(num_devices=4)
    meshes = self.manager.meshes

    self.assertEqual(self.manager.type, IpuTargetType.IPU)
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
    # Use IPUs 2 and 3, to make sure ids/indexes are handled properly.
    self.manager = IpuDeviceMeshManager.create_ipu_manager({2, 3})
    manager = self.manager
    # Existing IPU mesh => find from hardware IDs.
    mesh = manager.find([2, 3])
    self.assertEqual(manager.type, IpuTargetType.IPU)
    self.assertEqual(mesh.info.ipu_ids, [2, 3])
    self.assertEqual(mesh.local_device_index, 2)
    self.assertIn(2, mesh)
    self.assertNotIn(0, mesh)
    # No mesh with [0, 2]
    with self.assertRaises(IndexError):
      manager.find([0, 2])
    # Single IPU mesh
    mesh = manager.find(3)
    self.assertEqual(manager.type, IpuTargetType.IPU)
    self.assertEqual(mesh.info.ipu_ids, [3])
    self.assertEqual(mesh.local_device_index, 1)
    self.assertIn(3, mesh)
    # From XLA device_assignment: using local index directly.
    device_assignment = xla_client.DeviceAssignment.create([[1], [0]])
    mesh = manager.find(device_assignment)
    self.assertEqual(manager.type, IpuTargetType.IPU)
    self.assertEqual(mesh.info.ipu_ids, [2, 3])
    self.assertEqual(mesh.local_device_index, 2)

  def testIpuDeviceMeshManager__IpuDeviceMesh__Overlaps(self):
    self.manager = IpuDeviceMeshManager.create_ipu_manager({0, 1, 2, 3})
    mesh = self.manager.find([0, 1])
    self.assertTrue(mesh.overlaps(mesh))
    self.assertTrue(mesh.overlaps(self.manager.find([0, 1, 2, 3])))
    self.assertFalse(mesh.overlaps(self.manager.find([2, 3])))
    self.assertTrue(mesh.overlaps(self.manager.find(0)))

  def testIpuDeviceMeshManager__IpuDeviceMesh__OverlappingMeshIds(self):
    self.manager = IpuDeviceMeshManager.create_ipu_manager({0, 1, 2, 3})
    meshes = self.manager.meshes
    # Single IPU overlap?
    self.assertEqual(
        self.manager.overlapping_mesh_ids(0), [m.id for m in meshes[1:] if 0 in m]
    )
    # Multi IPUs overlap?
    self.assertEqual(
        self.manager.overlapping_mesh_ids(meshes[-1].id), [m.id for m in meshes[:-1]]
    )
    # Last mesh should overlap with everyone else!
    last_mesh_id = meshes[-1].id
    for m in meshes[:-1]:
      self.assertIn(last_mesh_id, self.manager.overlapping_mesh_ids(m.id))

  def testIpuDeviceMeshManager__create_ipu_device_mesh_manager__ipu_hardware(self):
    ipu_options = IpuPjRtOptions(use_ipu_model=False, num_devices=2)
    self.manager = create_ipu_device_mesh_manager(ipu_options)
    ipu_manager = self.manager

    self.assertIsInstance(ipu_manager, IpuDeviceMeshManager)
    self.assertEqual(ipu_manager.type, IpuTargetType.IPU)
    self.assertGreaterEqual(len(ipu_manager), 1)
    self.assertEqual(ipu_manager[0].type, IpuTargetType.IPU)
    self.assertEqual(ipu_manager[0].num_tiles_per_ipu, 1472)
    self.assertEqual(ipu_manager[0].version, "ipu2")

  def testIpuDeviceMeshManager__default_mesh__multi_ipus_hardware(self):
    self.manager = create_ipu_device_mesh_manager(
        IpuPjRtOptions(use_ipu_model=False, num_devices=4)
    )
    # First IPU mesh of size 1
    m = self.manager.default_mesh(1)
    assert m.size == 1
    assert m.id == 0
    # First IPU mesh of size 2
    m = self.manager.default_mesh(2)
    assert m.size == 2
    assert m.id == min([m.id for m in self.manager.meshes if m.size == 2])

  def testIpuDeviceMeshManager__from_mesh_id_to_index__proper_result(self):
    self.manager = create_ipu_device_mesh_manager(
        IpuPjRtOptions(use_ipu_model=False, num_devices=4)
    )
    # Consistent mapping.
    expected_indexes = [m.local_device_index for m in self.manager]
    indexes = [self.manager.from_mesh_id_to_index(m.id) for m in self.manager]
    self.assertEqual(indexes, expected_indexes)

  def testIpuDeviceMeshManager__count__ipu_hardware_different_sizes(self):
    self.manager = create_ipu_device_mesh_manager(
        IpuPjRtOptions(use_ipu_model=False, num_devices=4)
    )
    # Initial count for single IPU mesh.
    size, count = 1, self.manager.count(1)
    while (self.manager.count(size * 2)):
      # Number of mesh of size 2x should be half.
      self.assertEqual(self.manager.count(size * 2), count // 2)
      size, count = 2 * size, self.manager.count(2 * size)

  def testIpuDeviceMeshManager__attach__no_overlapping_logic(self):
    self.manager = IpuDeviceMeshManager.create_ipu_manager(num_devices=4)
    self.manager.detach_all()
    single_ipus = self.manager.count(1)
    # Should be able to attach all single IPU devices.
    for idx in range(single_ipus):
      ipu_id = self.manager.meshes[idx].local_hardware_id
      self.assertTrue(self.manager.attach(ipu_id, force_detach_overlapping=False))
      self.assertTrue(self.manager.is_attached(ipu_id))
    # Fail to attach all multi-ipus meshes.
    for idx in range(single_ipus, len(self.manager)):
      mesh_id = self.manager.meshes[idx].local_hardware_id
      self.assertFalse(self.manager.attach(mesh_id, force_detach_overlapping=False))

    self.manager.detach_all()
    # Should be able to attach last one (i.e. all devices mesh).
    self.assertTrue(self.manager.attach(self.manager.meshes[-1].local_hardware_id))
    self.assertTrue(self.manager.is_attached(self.manager.meshes[-1].local_hardware_id))
    # Fail to attach any other IPU mesh!
    for idx in range(0, len(self.manager) - 1):
      mesh_id = self.manager.meshes[idx].local_hardware_id
      self.assertFalse(self.manager.attach(mesh_id, force_detach_overlapping=False))

  def testIpuDeviceMeshManager__attach__overlapping_logic(self):
    self.manager = IpuDeviceMeshManager.create_ipu_manager(num_devices=4)
    self.manager.detach_all()
    manager = self.manager

    # Should be able to attach all IPU meshes.
    for idx in range(len(manager)):
      ipu_id = self.manager.meshes[idx].local_hardware_id
      self.assertTrue(manager.attach(ipu_id, force_detach_overlapping=True))
      self.assertTrue(manager.is_attached(ipu_id))
    # None attached, except last one.
    for idx in range(len(manager) - 1):
      ipu_id = self.manager.meshes[idx].local_hardware_id
      self.assertFalse(manager.is_attached(ipu_id))
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

  def testIpuDeviceMeshManager__create_ipu_manager__default_num_ipus(self):
    self.manager = IpuDeviceMeshManager.create_ipu_manager()
    self.assertEqual(self.manager.count(1), num_ipu_hw_available)
    self.assertEqual({
        m.is_attached for m in self.manager.meshes[:num_ipu_hw_available]
    }, {True})

  def testIpuDeviceMeshManager__create_ipu_manager__unsupported_number_ipus(self):
    with self.assertRaises(xla_extension.XlaRuntimeError):
      IpuDeviceMeshManager.create_ipu_manager(num_ipu_hw_available + 1)
    with self.assertRaises(xla_extension.XlaRuntimeError):
      IpuDeviceMeshManager.create_ipu_manager(3)
    with self.assertRaises(xla_extension.XlaRuntimeError):
      IpuDeviceMeshManager.create_ipu_manager(0)

  def testIpuDeviceMeshManager__create_ipu_manager__multi_sessions(self):
    manager0 = IpuDeviceMeshManager.create_ipu_manager(num_devices=2)
    # 2 single IPUs + single multi IPUs mesh.
    self.assertEqual(len(manager0), 2 + 1)
    self.assertEqual([m.id for m in manager0.meshes[:-1]], [0, 1])
    self.assertEqual([m.local_device_index for m in manager0.meshes], [0, 1, 2])
    self.assertEqual([m.is_attached for m in manager0.meshes], [True, True, False])
    # Another manager => should give another set of IPUs!
    manager1 = IpuDeviceMeshManager.create_ipu_manager(num_devices=2)
    self.assertEqual(len(manager1), 2 + 1)
    self.assertEqual([m.id for m in manager1.meshes[:-1]], [2, 3])
    self.assertEqual([m.local_device_index for m in manager0.meshes], [0, 1, 2])
    self.assertEqual([m.is_attached for m in manager1.meshes], [True, True, False])

    manager0.clear()
    # Try again with single IPU => should give 0.
    manager0 = IpuDeviceMeshManager.create_ipu_manager(num_devices=1)
    self.assertEqual(len(manager0), 1)
    self.assertEqual([m.id for m in manager0.meshes], [0])
    self.assertEqual([m.local_device_index for m in manager0.meshes], [0])
    self.assertEqual([m.is_attached for m in manager0.meshes], [True])

    manager0.clear()
    manager1.clear()

  def testIpuDeviceMeshManager__create_ipu_manager__exhaust_ipu_available(self):
    managers = []
    for _ in range(num_ipu_hw_available // 2):
      managers.append(IpuDeviceMeshManager.create_ipu_manager(num_devices=2))
    # All IPUs allocated/attached!
    with self.assertRaises(xla_extension.XlaRuntimeError):
      IpuDeviceMeshManager.create_ipu_manager(num_devices=2)
    for m in managers:
      m.clear()

  def testIpuDeviceMeshManager__create_ipu_manager__invalid_visible_devices(self):
    with self.assertRaises(xla_extension.XlaRuntimeError):
      IpuDeviceMeshManager.create_ipu_manager(set())
    with self.assertRaises(xla_extension.XlaRuntimeError):
      IpuDeviceMeshManager.create_ipu_manager({-1, 0})
    with self.assertRaises(xla_extension.XlaRuntimeError):
      IpuDeviceMeshManager.create_ipu_manager({0, num_ipu_hw_available})

  def testIpuDeviceMeshManager__create_ipu_manager__visible_devices(self):
    self.manager = IpuDeviceMeshManager.create_ipu_manager({0, 2, 3})
    self.assertEqual(len(self.manager), 4)
    self.assertEqual([m.id for m in self.manager.meshes[:-1]], [0, 2, 3])
    self.assertEqual([m.local_device_index for m in self.manager.meshes], [0, 1, 2, 3])
    self.assertEqual([m.is_attached for m in self.manager.meshes],
                     [True, True, True, False])
    # Another session with single IPU.
    manager1 = IpuDeviceMeshManager.create_ipu_manager(1)
    self.assertEqual(len(manager1), 1)
    self.assertEqual([m.id for m in manager1.meshes], [1])

  def testIpuDeviceMeshManager__create_ipu_device_mesh_manager__visible_devices(self):
    ipu_options = IpuPjRtOptions(visible_devices={0, 2, 3}, use_ipu_model=False)
    self.manager = create_ipu_device_mesh_manager(ipu_options)
    self.assertEqual(len(self.manager), 4)
    self.assertEqual([m.id for m in self.manager.meshes[:-1]], [0, 2, 3])
    self.assertEqual([m.local_device_index for m in self.manager.meshes], [0, 1, 2, 3])
    self.assertEqual([m.is_attached for m in self.manager.meshes],
                     [True, True, True, False])


if __name__ == "__main__":
  absltest.main()
